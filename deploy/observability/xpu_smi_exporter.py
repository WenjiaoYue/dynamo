#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Intel XPU-SMI Prometheus Exporter.

Collects Intel GPU metrics via xpu-smi and exposes them in Prometheus format
on a configurable HTTP port (default: 9966).

Usage:
    python xpu_smi_exporter.py [--port 9966] [--interval 5]

Metrics exposed (matching the Grafana dashboard xpu-smi-metrics.json):
    xpu_power_watts              - GPU power draw in watts
    xpu_frequency_mhz            - GPU core frequency in MHz
    xpu_memory_used_bytes        - GPU memory used in bytes
    xpu_memory_free_bytes        - GPU memory free in bytes
    xpu_memory_utilization_ratio - GPU memory utilization (0-1)
    xpu_temperature_celsius      - GPU temperature in Celsius (location="gpu" or location="memory")
    xpu_pcie_read_bytes_total    - PCIe read throughput (counter, bytes)
    xpu_pcie_write_bytes_total   - PCIe write throughput (counter, bytes)
    xpu_engine_group_compute_engine_util - Compute engine utilization %
    xpu_engine_group_render_engine_util  - Render engine utilization %
    xpu_engine_group_copy_engine_util    - Copy engine utilization %
    xpu_memory_read_bytes_total  - Memory read throughput (counter, bytes)
    xpu_memory_write_bytes_total - Memory write throughput (counter, bytes)
"""

import argparse
import json
import logging
import subprocess
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("xpu-smi-exporter")

# xpu-smi dump metric IDs
# 0=GPU Util%, 1=Power(W), 2=Freq(MHz), 3=CoreTemp(C), 4=MemTemp(C),
# 5=MemUtil%, 6=MemRead(kB/s), 7=MemWrite(kB/s), 18=MemUsed(MiB),
# 19=PCIeRead(kB/s), 20=PCIeWrite(kB/s),
# 31=ComputeEngGrp%, 32=RenderEngGrp%, 33=MediaEngGrp%, 34=CopyEngGrp%
DUMP_METRICS = "0,1,2,3,4,5,6,7,18,19,20,31,32,33,34"

# Metric name in dump header -> (prometheus_name, help, type, unit_conversion, location)
# unit_conversion: multiply raw value by this factor
# location: optional label value for the "location" label (used for temperature metrics)
DUMP_HEADER_MAP = {
    "GPU Utilization (%)": ("xpu_gpu_utilization_percent", "GPU utilization percentage", "gauge", 1, None),
    "GPU Power (W)": ("xpu_power_watts", "GPU power consumption in watts", "gauge", 1, None),
    "GPU Frequency (MHz)": ("xpu_frequency_mhz", "GPU core frequency in MHz", "gauge", 1, None),
    "GPU Core Temperature (Celsius Degree)": ("xpu_temperature_celsius", "GPU core temperature in Celsius", "gauge", 1, "gpu"),
    "GPU Memory Temperature (Celsius Degree)": ("xpu_temperature_celsius", "GPU memory temperature in Celsius", "gauge", 1, "memory"),
    "GPU Memory Utilization (%)": ("xpu_memory_utilization_percent", "GPU memory utilization percentage", "gauge", 1, None),
    "GPU Memory Read (kB/s)": ("xpu_memory_read_bytes_total", "GPU memory read throughput in bytes per second", "gauge", 1024, None),
    "GPU Memory Write (kB/s)": ("xpu_memory_write_bytes_total", "GPU memory write throughput in bytes per second", "gauge", 1024, None),
    "GPU Memory Used (MiB)": ("xpu_memory_used_bytes", "GPU memory used in bytes", "gauge", 1048576, None),
    "PCIe Read (kB/s)": ("xpu_pcie_read_bytes_total", "PCIe read throughput in bytes per second", "gauge", 1024, None),
    "PCIe Write (kB/s)": ("xpu_pcie_write_bytes_total", "PCIe write throughput in bytes per second", "gauge", 1024, None),
    "Compute engine group utilization (%)": ("xpu_engine_group_compute_engine_util", "Compute engine group utilization percentage", "gauge", 1, None),
    "Render engine group utilization (%)": ("xpu_engine_group_render_engine_util", "Render engine group utilization percentage", "gauge", 1, None),
    "Media engine group utilization (%)": ("xpu_engine_group_media_engine_util", "Media engine group utilization percentage", "gauge", 1, None),
    "Copy engine group utilization (%)": ("xpu_engine_group_copy_engine_util", "Copy engine group utilization percentage", "gauge", 1, None),
}


class MetricsCollector:
    """Collects XPU metrics from xpu-smi commands.

    Runs a background thread that periodically calls xpu-smi and caches the
    results.  The /metrics handler returns the cached snapshot instantly,
    avoiding Prometheus scrape-timeout issues caused by slow xpu-smi calls.
    """

    def __init__(self, interval: int = 5):
        self._lock = threading.Lock()
        self._metrics: dict = {}
        self._devices: list = []
        self._device_memory_total: dict = {}  # device_id -> total memory bytes
        self._interval = interval
        self._discover_devices()

    def _discover_devices(self):
        """Discover available XPU devices."""
        try:
            result = subprocess.run(
                ["xpu-smi", "discovery", "-j"],
                capture_output=True, text=True, timeout=10,
            )
            data = json.loads(result.stdout)
            self._devices = [d["device_id"] for d in data.get("device_list", [])]
            # Get total memory per device
            for dev_id in self._devices:
                self._get_device_memory_total(dev_id)
            logger.info(f"Discovered {len(self._devices)} XPU device(s): {self._devices}")
        except Exception as e:
            logger.error(f"Failed to discover devices: {e}")
            self._devices = []

    def _get_device_memory_total(self, device_id: int):
        """Get total physical memory for a device."""
        try:
            result = subprocess.run(
                ["xpu-smi", "discovery", "-d", str(device_id), "-j"],
                capture_output=True, text=True, timeout=10,
            )
            data = json.loads(result.stdout)
            total = int(data.get("memory_physical_size_byte", 0))
            self._device_memory_total[device_id] = total
            logger.info(f"Device {device_id}: total memory = {total / (1024**3):.1f} GiB")
        except Exception as e:
            logger.warning(f"Failed to get memory total for device {device_id}: {e}")

    def _collect_dump_metrics(self, device_id: int) -> dict:
        """Collect metrics via xpu-smi dump for a single device."""
        metrics = {}
        try:
            result = subprocess.run(
                ["xpu-smi", "dump", "-d", str(device_id), "-m", DUMP_METRICS, "-n", "1"],
                capture_output=True, text=True, timeout=15,
            )
            lines = result.stdout.strip().split("\n")
            if len(lines) < 2:
                return metrics

            # Parse header
            header_line = lines[0]
            headers = [h.strip() for h in header_line.split(",")]
            # Parse data (last line)
            data_line = lines[-1]
            values = [v.strip() for v in data_line.split(",")]

            if len(headers) != len(values):
                logger.warning(f"Header/value count mismatch: {len(headers)} vs {len(values)}")
                return metrics

            # Skip Timestamp and DeviceId columns
            for i in range(2, len(headers)):
                header = headers[i]
                raw_val = values[i]

                if raw_val == "N/A" or raw_val == "":
                    continue

                mapping = DUMP_HEADER_MAP.get(header)
                if not mapping:
                    continue

                prom_name, help_text, metric_type, conversion, location = mapping
                try:
                    val = float(raw_val) * conversion
                    labels = {"device_id": str(device_id)}
                    if location is not None:
                        labels["location"] = location
                    # Use a unique key to avoid overwriting when two headers map to
                    # the same prometheus metric name (e.g. both temperature entries
                    # map to xpu_temperature_celsius but differ by location label).
                    metric_key = prom_name if location is None else f"{prom_name}_{location}"
                    metrics[metric_key] = {
                        "prom_name": prom_name,
                        "value": val,
                        "help": help_text,
                        "type": metric_type,
                        "labels": labels,
                    }
                except ValueError:
                    continue

        except subprocess.TimeoutExpired:
            logger.warning(f"xpu-smi dump timed out for device {device_id}")
        except Exception as e:
            logger.warning(f"Error collecting dump metrics for device {device_id}: {e}")
        return metrics

    def _collect_stats_metrics(self, device_id: int) -> dict:
        """Collect metrics via xpu-smi stats for a single device (fallback/supplement)."""
        metrics = {}
        try:
            result = subprocess.run(
                ["xpu-smi", "stats", "-d", str(device_id), "-j"],
                capture_output=True, text=True, timeout=10,
            )
            data = json.loads(result.stdout)
            labels = {"device_id": str(device_id)}

            # Device-level metrics
            for entry in data.get("device_level", []):
                mtype = entry.get("metrics_type", "")
                val = entry.get("value")
                if val is None:
                    continue
                if mtype == "XPUM_STATS_POWER":
                    metrics["xpu_power_watts"] = {
                        "prom_name": "xpu_power_watts",
                        "value": float(val),
                        "help": "GPU power consumption in watts",
                        "type": "gauge",
                        "labels": labels,
                    }

            # Tile-level metrics (aggregate to device level)
            tile_data = data.get("tile_level", [])
            if tile_data:
                mem_used_sum = 0.0
                mem_util_sum = 0.0
                freq_sum = 0.0
                tile_count = 0
                for tile in tile_data:
                    tile_count += 1
                    for entry in tile.get("data_list", []):
                        mtype = entry.get("metrics_type", "")
                        val = entry.get("value")
                        if val is None:
                            continue
                        if mtype == "XPUM_STATS_MEMORY_USED":
                            mem_used_sum += float(val)  # MiB
                        elif mtype == "XPUM_STATS_MEMORY_UTILIZATION":
                            mem_util_sum += float(val)
                        elif mtype == "XPUM_STATS_GPU_FREQUENCY":
                            freq_sum += float(val)

                if tile_count > 0:
                    # Memory used: sum across tiles, convert MiB -> bytes
                    mem_used_bytes = mem_used_sum * 1048576
                    metrics["xpu_memory_used_bytes"] = {
                        "prom_name": "xpu_memory_used_bytes",
                        "value": mem_used_bytes,
                        "help": "GPU memory used in bytes",
                        "type": "gauge",
                        "labels": labels,
                    }
                    # Memory free: total - used
                    total = self._device_memory_total.get(device_id, 0)
                    if total > 0:
                        metrics["xpu_memory_free_bytes"] = {
                            "prom_name": "xpu_memory_free_bytes",
                            "value": max(0, total - mem_used_bytes),
                            "help": "GPU memory free in bytes",
                            "type": "gauge",
                            "labels": labels,
                        }
                    # Average frequency across tiles
                    metrics["xpu_frequency_mhz"] = {
                        "prom_name": "xpu_frequency_mhz",
                        "value": freq_sum / tile_count,
                        "help": "GPU core frequency in MHz",
                        "type": "gauge",
                        "labels": labels,
                    }

        except subprocess.TimeoutExpired:
            logger.warning(f"xpu-smi stats timed out for device {device_id}")
        except Exception as e:
            logger.warning(f"Error collecting stats for device {device_id}: {e}")
        return metrics

    def start_background_collection(self):
        """Start a daemon thread that collects metrics periodically."""
        def _loop():
            while True:
                try:
                    self.collect()
                except Exception as e:
                    logger.error(f"Background collection error: {e}")
                time.sleep(self._interval)
        t = threading.Thread(target=_loop, daemon=True)
        t.start()
        logger.info(f"Background collection started (interval={self._interval}s)")

    def collect(self):
        """Collect all metrics from all devices."""
        all_metrics = {}
        for dev_id in self._devices:
            # Collect from dump first
            dump_metrics = self._collect_dump_metrics(dev_id)
            # Collect from stats (supplements dump, especially for memory)
            stats_metrics = self._collect_stats_metrics(dev_id)

            # Merge: dump takes priority for metrics it provides,
            # stats fills in what dump doesn't have
            merged = {**stats_metrics, **dump_metrics}
            # But for memory_used_bytes and memory_free_bytes, prefer stats
            # since dump often returns N/A for memory
            if "xpu_memory_used_bytes" in stats_metrics:
                merged["xpu_memory_used_bytes"] = stats_metrics["xpu_memory_used_bytes"]
            if "xpu_memory_free_bytes" in stats_metrics:
                merged["xpu_memory_free_bytes"] = stats_metrics["xpu_memory_free_bytes"]

            for key, data in merged.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(data)

        with self._lock:
            self._metrics = all_metrics

    def format_prometheus(self) -> str:
        """Format collected metrics in Prometheus exposition format."""
        with self._lock:
            metrics = self._metrics.copy()

        lines = []
        # Group by prometheus metric name to emit # HELP / # TYPE once per metric
        seen_prom_names = set()
        for metric_key, entries in sorted(metrics.items()):
            if not entries:
                continue
            first = entries[0]
            prom_name = first.get("prom_name", metric_key)
            if prom_name not in seen_prom_names:
                lines.append(f"# HELP {prom_name} {first['help']}")
                lines.append(f"# TYPE {prom_name} {first['type']}")
                seen_prom_names.add(prom_name)
            for entry in entries:
                label_parts = ",".join(
                    f'{k}="{v}"' for k, v in sorted(entry["labels"].items())
                )
                lines.append(f"{prom_name}{{{label_parts}}} {entry['value']}")
        lines.append("")
        return "\n".join(lines)


class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler for /metrics endpoint."""

    collector: MetricsCollector = None  # Set by main

    def do_GET(self):
        try:
            if self.path == "/metrics" or self.path == "/":
                output = self.collector.format_prometheus()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                self.end_headers()
                self.wfile.write(output.encode("utf-8"))
            elif self.path == "/healthz":
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"ok\n")
            else:
                self.send_response(404)
                self.end_headers()
        except BrokenPipeError:
            pass

    def log_message(self, format, *args):
        # Suppress per-request logging to reduce noise
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Intel XPU-SMI Prometheus Exporter"
    )
    parser.add_argument(
        "--port", type=int, default=9966,
        help="Port to expose Prometheus metrics on (default: 9966)",
    )
    parser.add_argument(
        "--interval", type=int, default=5,
        help="Seconds between background metric collections (default: 5)",
    )
    args = parser.parse_args()

    collector = MetricsCollector(interval=args.interval)
    if not collector._devices:
        logger.error("No XPU devices found. Exiting.")
        sys.exit(1)

    # Do an initial collection to verify it works
    collector.collect()
    initial = collector.format_prometheus()
    logger.info(f"Initial collection complete, {len(initial)} bytes of metrics")

    # Start background collection so /metrics returns cached data instantly
    collector.start_background_collection()

    MetricsHandler.collector = collector

    server = HTTPServer(("0.0.0.0", args.port), MetricsHandler)
    logger.info(f"Serving XPU metrics on http://0.0.0.0:{args.port}/metrics")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down exporter")
        server.shutdown()


if __name__ == "__main__":
    main()
