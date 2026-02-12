# Common Issues

## Installation Issues

### Database Connection Error

**Problem**: Cannot connect to PostgreSQL

**Solution**:
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Verify connection settings in .env
DATABASE_URL=postgresql://user:pass@localhost:5432/aegispcap
```

### Redis Connection Error

**Problem**: Cannot connect to Redis

**Solution**:
```bash
# Start Redis
sudo systemctl start redis

# Test connection
redis-cli ping
```

## Runtime Issues

### PCAP Processing Fails

**Problem**: Error processing PCAP file

**Solution**:
- Verify PCAP file is valid
- Check file permissions
- Review logs: `tail -f logs/aegispcap.log`

### High Memory Usage

**Problem**: System using too much memory

**Solution**:
- Reduce batch size in configuration
- Enable flow caching
- Scale horizontally with Kubernetes

See [FAQ](faq.md) for more questions.
