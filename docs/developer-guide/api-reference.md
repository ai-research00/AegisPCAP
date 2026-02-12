# API Reference

## REST API Endpoints

### Flow Analysis

```
POST /api/v1/pcap/upload
GET /api/v1/flows
GET /api/v1/flows/{flow_id}
```

### Threat Detection

```
GET /api/v1/alerts
GET /api/v1/alerts/{alert_id}
POST /api/v1/alerts/{alert_id}/acknowledge
```

### Community Features

```
GET /api/v1/plugins
POST /api/v1/plugins/install
GET /api/v1/models
POST /api/v1/models/upload
```

## WebSocket API

```
ws://localhost:8000/ws/alerts
```

See [Plugin Development](plugin-development.md) for extending the API.
