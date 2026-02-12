#!/usr/bin/env python3
"""
AegisPCAP - AI-Agent-Driven Network Traffic Intelligence
Main entry point and CLI interface
"""
import logging
import argparse
import sys
from pathlib import Path
import json

from pipeline import AnalysisPipeline
from config import *

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main CLI entry point"""
    
    parser = argparse.ArgumentParser(
        description="AegisPCAP - AI-Agent-Driven Network Traffic Intelligence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a PCAP file
  python main.py analyze data/sample.pcap
  
  # Analyze with a specific query
  python main.py analyze data/sample.pcap -q "find C2 communications"
  
  # Train models (requires labeled data)
  python main.py train --data labeled_flows.csv
  
  # Start API server
  python main.py server --host 0.0.0.0 --port 8000
  
  # Run tests
  python main.py test
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze PCAP file')
    analyze_parser.add_argument('pcap_path', help='Path to PCAP file')
    analyze_parser.add_argument('-q', '--query', help='Analysis query')
    analyze_parser.add_argument('-o', '--output', help='Output JSON file')
    analyze_parser.add_argument('--format', choices=['json', 'html', 'pdf'], default='json',
                               help='Output format')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train ML models')
    train_parser.add_argument('--data', help='Training data file')
    train_parser.add_argument('--labels', help='Labels file')
    train_parser.add_argument('--model', choices=['ensemble', 'c2', 'exfil', 'botnet', 'all'],
                             default='all', help='Model to train')
    
    # Server command
    server_parser = subparsers.add_parser('server', help='Start API server')
    server_parser.add_argument('--host', default='0.0.0.0', help='Server host')
    server_parser.add_argument('--port', type=int, default=API_PORT, help='Server port')
    server_parser.add_argument('--workers', type=int, default=API_WORKERS, help='Number of workers')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run tests')
    test_parser.add_argument('--module', help='Specific test module')
    test_parser.add_argument('--coverage', action='store_true', help='Show coverage report')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Manage configuration')
    config_parser.add_argument('--show', action='store_true', help='Show current configuration')
    config_parser.add_argument('--validate', action='store_true', help='Validate configuration')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute commands
    if args.command == 'analyze':
        analyze_pcap(args)
    
    elif args.command == 'train':
        train_models(args)
    
    elif args.command == 'server':
        start_server(args)
    
    elif args.command == 'test':
        run_tests(args)
    
    elif args.command == 'config':
        manage_config(args)


def analyze_pcap(args):
    """Execute PCAP analysis"""
    
    pcap_path = Path(args.pcap_path)
    
    if not pcap_path.exists():
        logger.error(f"PCAP file not found: {pcap_path}")
        sys.exit(1)
    
    logger.info(f"Starting analysis of {pcap_path}")
    
    # Initialize pipeline
    pipeline = AnalysisPipeline()
    
    # Run analysis
    results = pipeline.analyze_pcap(str(pcap_path), query=args.query)
    
    # Output results
    if args.output:
        output_file = Path(args.output)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_file}")
    else:
        print(json.dumps(results, indent=2, default=str))
    
    # Print summary
    summary = results.get('summary', {})
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total Packets: {summary.get('total_packets', 0)}")
    print(f"Total Flows: {summary.get('total_flows', 0)}")
    print(f"High Risk Flows: {summary.get('high_risk_flows', 0)}")
    print(f"Medium Risk Flows: {summary.get('medium_risk_flows', 0)}")
    print(f"Overall Risk Level: {summary.get('overall_risk_level', 'unknown')}")
    print("="*60)


def train_models(args):
    """Train ML models"""
    
    logger.info(f"Training {args.model} model(s)")
    
    pipeline = AnalysisPipeline()
    pipeline.train_models(training_data=args.data)
    
    logger.info("Training complete")


def start_server(args):
    """Start REST API server"""
    
    logger.info(f"Starting API server on {args.host}:{args.port}")
    
    try:
        import uvicorn
        from src.api.main import app  # Assuming API module exists
        
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            workers=args.workers,
            log_level=LOG_LEVEL.lower()
        )
    
    except ImportError:
        logger.error("API module not found. Please implement src/api/main.py")
        sys.exit(1)


def run_tests(args):
    """Run test suite"""
    
    logger.info("Running tests")
    
    try:
        import pytest
        
        if args.module:
            pytest.main([f'tests/{args.module}', '-v'])
        else:
            pytest.main(['tests/', '-v', '--cov=src' if args.coverage else ''])
    
    except ImportError:
        logger.error("pytest not found. Install with: pip install pytest pytest-cov")
        sys.exit(1)


def manage_config(args):
    """Manage configuration"""
    
    if args.show:
        print("AegisPCAP Configuration")
        print("="*60)
        print(f"Project: {PROJECT_NAME} v{VERSION}")
        print(f"Database: {DB_HOST}:{DB_PORT}/{DB_NAME}")
        print(f"Redis: {REDIS_HOST}:{REDIS_PORT}")
        print(f"Elasticsearch: {ES_HOST}:{ES_PORT}")
        print(f"API: {API_HOST}:{API_PORT}")
        print(f"Log Level: {LOG_LEVEL}")
        print(f"GDPR Mode: {GDPR_MODE}")
        print(f"Feature Groups: {FEATURE_GROUPS}")
        print("="*60)
    
    elif args.validate:
        # Validate critical settings
        issues = []
        
        if not DB_HOST or not DB_USER:
            issues.append("Database credentials not configured")
        
        if not FEATURE_GROUPS:
            issues.append("No feature groups enabled")
        
        if issues:
            print("Configuration Issues:")
            for issue in issues:
                print(f"  ⚠️ {issue}")
        else:
            print("✅ Configuration is valid")


if __name__ == "__main__":
    main()
