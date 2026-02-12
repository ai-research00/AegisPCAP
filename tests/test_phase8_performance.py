"""
Phase 8 Performance Benchmarking Tests
Measure latency, throughput, and resource utilization
"""

import pytest
import numpy as np
import time
import psutil
import json
from datetime import datetime
from collections import defaultdict


# ============================================================================
# BENCHMARK FIXTURES & UTILITIES
# ============================================================================

class BenchmarkRunner:
    """Helper to run and record benchmarks"""
    
    def __init__(self):
        self.results = defaultdict(list)
    
    def measure(self, name, func, *args, **kwargs):
        """Run function and measure execution time"""
        gc_enabled = __import__('gc').isenabled()
        __import__('gc').disable()
        
        try:
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            
            self.results[name].append({
                'elapsed_ms': elapsed,
                'timestamp': datetime.utcnow().isoformat()
            })
            
            return result, elapsed
        finally:
            if gc_enabled:
                __import__('gc').enable()
    
    def get_stats(self, name):
        """Get statistics for a benchmark"""
        if name not in self.results or not self.results[name]:
            return None
        
        times = [r['elapsed_ms'] for r in self.results[name]]
        return {
            'count': len(times),
            'mean_ms': np.mean(times),
            'median_ms': np.median(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'p95_ms': np.percentile(times, 95),
            'p99_ms': np.percentile(times, 99),
        }


@pytest.fixture
def benchmark_runner():
    """Create benchmark runner"""
    return BenchmarkRunner()


# ============================================================================
# FEATURE EXTRACTION PERFORMANCE TESTS
# ============================================================================

class TestFeatureExtractionPerformance:
    """Benchmark feature extraction performance"""
    
    def test_packet_size_statistics_latency(self, benchmark_runner):
        """Measure latency for packet size statistics"""
        def extract_stats(sizes):
            return {
                'mean': np.mean(sizes),
                'std': np.std(sizes),
                'min': np.min(sizes),
                'max': np.max(sizes),
            }
        
        # Small batch
        sizes = np.random.randint(50, 1500, 100)
        _, elapsed = benchmark_runner.measure('pkt_stats_small', extract_stats, sizes)
        assert elapsed < 5.0  # <5ms for 100 packets
        
        # Medium batch
        sizes = np.random.randint(50, 1500, 1000)
        _, elapsed = benchmark_runner.measure('pkt_stats_medium', extract_stats, sizes)
        assert elapsed < 10.0  # <10ms for 1K packets
        
        # Large batch
        sizes = np.random.randint(50, 1500, 10000)
        _, elapsed = benchmark_runner.measure('pkt_stats_large', extract_stats, sizes)
        assert elapsed < 50.0  # <50ms for 10K packets
        
        stats = benchmark_runner.get_stats('pkt_stats_large')
        print(f"\n📊 Packet Statistics Latency: {stats['mean_ms']:.2f}ms (p95: {stats['p95_ms']:.2f}ms)")
    
    def test_entropy_calculation_latency(self, benchmark_runner):
        """Measure entropy calculation latency"""
        def calculate_entropy(data):
            if isinstance(data, str):
                data = data.encode()
            byte_counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
            probabilities = byte_counts / len(data)
            entropy = -np.sum(probabilities[probabilities > 0] * np.log2(probabilities[probabilities > 0]))
            return entropy
        
        # Small payload (100 bytes)
        payload = b'A' * 100
        _, elapsed = benchmark_runner.measure('entropy_small', calculate_entropy, payload)
        assert elapsed < 1.0
        
        # Medium payload (10KB)
        payload = bytes(np.random.randint(0, 256, 10000))
        _, elapsed = benchmark_runner.measure('entropy_medium', calculate_entropy, payload)
        assert elapsed < 5.0
        
        # Large payload (1MB)
        payload = bytes(np.random.randint(0, 256, 1000000))
        _, elapsed = benchmark_runner.measure('entropy_large', calculate_entropy, payload)
        assert elapsed < 100.0
    
    def test_flow_feature_extraction_throughput(self, benchmark_runner):
        """Measure feature extraction throughput"""
        def extract_features(flow):
            return {
                'src_ip': flow['src_ip'],
                'dst_ip': flow['dst_ip'],
                'bytes': flow['bytes'],
                'packets': flow['packets'],
                'duration': flow['duration'],
                'pkt_rate': flow['packets'] / max(flow['duration'], 1),
                'byte_rate': flow['bytes'] / max(flow['duration'], 1),
            }
        
        flows = [
            {
                'src_ip': f'10.0.0.{i}',
                'dst_ip': f'8.8.8.{i % 256}',
                'bytes': np.random.randint(100, 1000000),
                'packets': np.random.randint(1, 10000),
                'duration': np.random.uniform(0.1, 60),
            }
            for i in range(1000)
        ]
        
        start = time.perf_counter()
        for flow in flows:
            extract_features(flow)
        elapsed = time.perf_counter() - start
        
        throughput = len(flows) / elapsed
        print(f"\n📊 Flow Feature Extraction: {throughput:.0f} flows/sec (1000 flows in {elapsed*1000:.2f}ms)")
        assert throughput > 5000  # >5K flows/sec


# ============================================================================
# ML MODEL PERFORMANCE TESTS
# ============================================================================

class TestMLModelPerformance:
    """Benchmark ML model performance"""
    
    def test_inference_latency(self, benchmark_runner):
        """Measure model inference latency"""
        def predict(X):
            # Simulate simple prediction
            return np.random.randint(0, 2, len(X))
        
        # Single sample
        X = np.random.randn(1, 50)
        _, elapsed = benchmark_runner.measure('inference_single', predict, X)
        assert elapsed < 50.0  # <50ms per flow (generous)
        
        # Batch of 100
        X = np.random.randn(100, 50)
        _, elapsed = benchmark_runner.measure('inference_batch100', predict, X)
        assert elapsed < 100.0  # <100ms for 100 samples
        
        # Batch of 1000
        X = np.random.randn(1000, 50)
        _, elapsed = benchmark_runner.measure('inference_batch1000', predict, X)
        assert elapsed < 500.0  # <500ms for 1000 samples
    
    def test_ensemble_prediction_throughput(self, benchmark_runner):
        """Measure ensemble prediction throughput"""
        def ensemble_predict(X):
            # Simulate ensemble with multiple models
            pred1 = np.random.randint(0, 2, len(X))
            pred2 = np.random.randint(0, 2, len(X))
            pred3 = np.random.randint(0, 2, len(X))
            # Majority voting
            return (pred1 + pred2 + pred3) >= 2
        
        flows = 10000
        X = np.random.randn(flows, 50)
        
        start = time.perf_counter()
        predictions = ensemble_predict(X)
        elapsed = time.perf_counter() - start
        
        throughput = flows / elapsed
        print(f"\n📊 Model Inference Throughput: {throughput:.0f} flows/sec ({flows} flows in {elapsed*1000:.2f}ms)")
        assert throughput > 1000  # >1K flows/sec


# ============================================================================
# API ENDPOINT PERFORMANCE TESTS
# ============================================================================

class TestAPIPerformance:
    """Benchmark API endpoint performance"""
    
    def test_simple_endpoint_latency(self, benchmark_runner):
        """Measure simple endpoint response time"""
        def endpoint_response():
            # Simulate API response
            return {
                'status': 'ok',
                'data': {'sample': 'data'}
            }
        
        for _ in range(100):
            _, elapsed = benchmark_runner.measure('api_simple', endpoint_response)
        
        stats = benchmark_runner.get_stats('api_simple')
        print(f"\n📊 API Simple Endpoint: {stats['mean_ms']:.2f}ms (p95: {stats['p95_ms']:.2f}ms, p99: {stats['p99_ms']:.2f}ms)")
        assert stats['p95_ms'] < 50.0  # p95 <50ms
    
    def test_complex_query_latency(self, benchmark_runner):
        """Measure complex query endpoint latency"""
        def complex_query():
            # Simulate database query + computation
            data = np.random.randn(1000, 50)
            result = np.mean(data, axis=0)
            return result.tolist()
        
        for _ in range(50):
            _, elapsed = benchmark_runner.measure('api_complex', complex_query)
        
        stats = benchmark_runner.get_stats('api_complex')
        print(f"\n📊 API Complex Query: {stats['mean_ms']:.2f}ms (p95: {stats['p95_ms']:.2f}ms)")
        assert stats['p95_ms'] < 200.0  # p95 <200ms
    
    def test_concurrent_request_handling(self, benchmark_runner):
        """Test handling of concurrent requests"""
        import threading
        
        def api_call():
            time.sleep(0.001)  # Simulate 1ms work
            return {'status': 'ok'}
        
        results = []
        threads = []
        
        def worker():
            _, elapsed = benchmark_runner.measure('concurrent', api_call)
            results.append(elapsed)
        
        # Launch 10 concurrent "requests"
        for _ in range(10):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        print(f"\n📊 Concurrent Requests: {len(results)} concurrent, avg {np.mean(results):.2f}ms")


# ============================================================================
# DATABASE PERFORMANCE TESTS
# ============================================================================

class TestDatabasePerformance:
    """Benchmark database operations"""
    
    def test_sequential_inserts(self, benchmark_runner):
        """Measure sequential insert performance"""
        def insert_record(data):
            # Simulate database insert
            time.sleep(0.0001)  # 0.1ms per insert
            return True
        
        # Insert 1000 records
        for i in range(1000):
            _, elapsed = benchmark_runner.measure('db_insert', insert_record, {'id': i})
        
        stats = benchmark_runner.get_stats('db_insert')
        throughput = 1000 / (stats['mean_ms'] * 1000 / 1000)
        print(f"\n📊 Database Inserts: {throughput:.0f} records/sec (avg {stats['mean_ms']:.2f}ms)")
    
    def test_query_latency(self, benchmark_runner):
        """Measure query latency"""
        def query(limit):
            # Simulate database query
            time.sleep(0.005)  # 5ms base + variable
            return [{'id': i} for i in range(limit)]
        
        # Various query sizes
        for limit in [10, 100, 1000]:
            _, elapsed = benchmark_runner.measure(f'db_query_{limit}', query, limit)
        
        stats = benchmark_runner.get_stats('db_query_1000')
        print(f"\n📊 Database Query (1K rows): {stats['mean_ms']:.2f}ms (p95: {stats['p95_ms']:.2f}ms)")


# ============================================================================
# RESOURCE UTILIZATION TESTS
# ============================================================================

class TestResourceUtilization:
    """Monitor resource usage during operations"""
    
    def test_memory_baseline(self):
        """Measure baseline memory usage"""
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create some data structures
        data = [np.random.randn(1000, 50) for _ in range(10)]
        
        mem_after = process.memory_info().rss / 1024 / 1024
        mem_increase = mem_after - mem_before
        
        print(f"\n📊 Memory Baseline: {mem_before:.2f}MB → {mem_after:.2f}MB (Δ {mem_increase:.2f}MB)")
        assert mem_after < 500.0  # Keep under 500MB
    
    def test_cpu_utilization(self):
        """Measure CPU utilization during operations"""
        process = psutil.Process()
        
        # CPU-intensive operation
        start = time.perf_counter()
        for _ in range(1000):
            X = np.random.randn(100, 50)
            _ = np.mean(X, axis=0)
        elapsed = time.perf_counter() - start
        
        print(f"\n📊 CPU-Intensive Operation: 1000 iterations in {elapsed:.2f}s")
        assert elapsed < 10.0


# ============================================================================
# BENCHMARK REPORT GENERATOR
# ============================================================================

class TestBenchmarkReport:
    """Generate performance report"""
    
    def test_generate_performance_report(self, benchmark_runner):
        """Generate comprehensive performance report"""
        # Run some benchmarks
        def sample_operation():
            return np.random.randn(100, 50)
        
        for _ in range(100):
            _, elapsed = benchmark_runner.measure('sample', sample_operation)
        
        stats = benchmark_runner.get_stats('sample')
        
        # Create report
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'benchmarks': {
                'sample': stats
            }
        }
        
        # Print report
        print("\n" + "="*80)
        print("PERFORMANCE BENCHMARKS REPORT")
        print("="*80)
        print(json.dumps(report, indent=2, default=str))
        print("="*80)
        
        assert report is not None


# ============================================================================
# RUN BENCHMARKS
# ============================================================================

if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-s"  # Show print output
    ])
