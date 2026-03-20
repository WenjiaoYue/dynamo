# Dynamo Observability Stack

Configuration files for the Dynamo observability stack: metrics, tracing, logging, and visualization.

For full documentation, see [docs/observability/](../../docs/observability/).

## Quick Start

```bash
# Requires deploy/docker-compose.yml to be running first (for NATS and etcd)
docker compose -f deploy/docker-observability.yml up -d
```

## Services & Ports

| Service | Port(s) | Description |
|---------|---------|-------------|
| Grafana | 3000 | Dashboards & visualization (login: `dynamo` / `dynamo`) |
| Prometheus | 9090 | Metrics storage & query |
| Loki | 3100 | Log aggregation |
| Tempo | 3200 | Distributed tracing |
| OTel Collector | 4317 (gRPC), 4318 (HTTP) | OTLP ingestion point for traces & logs |
| DCGM Exporter | 9401 | NVIDIA GPU metrics |
| NATS Exporter | 7777 | NATS messaging metrics |

## Configuration Files

| File | Description |
|------|-------------|
| `prometheus.yml` | Prometheus scrape configuration — defines which services to collect metrics from |
| `otel-collector.yaml` | OpenTelemetry Collector pipeline — routes OTLP traces to Tempo and logs to Loki |
| `loki.yaml` | Loki log aggregation backend configuration |
| `tempo.yaml` | Tempo distributed tracing backend configuration |
| `grafana-datasources.yml` | Grafana Prometheus datasource provisioning |
| `tempo-datasource.yml` | Grafana Tempo datasource provisioning |
| `loki-datasource.yml` | Grafana Loki datasource provisioning |
| `grafana_dashboards/` | Pre-built Grafana dashboard JSON files |

## Grafana Dashboards

Pre-built dashboards are in `grafana_dashboards/`. See [`grafana_dashboards/README.md`](grafana_dashboards/README.md) for the full list.

## Kubernetes

For Kubernetes deployment, see [`k8s/MONITORING_SETUP.md`](k8s/MONITORING_SETUP.md).
