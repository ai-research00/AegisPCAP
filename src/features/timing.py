"""
Timing Features - Inter-arrival time analysis and beaconing detection
"""
import numpy as np
from typing import Dict, List
import logging
from scipy import signal

logger = logging.getLogger(__name__)


class TimingFeatures:
    """Extract timing-based anomaly detection features"""
    
    @staticmethod
    def extract_features(flow: Dict) -> Dict:
        """Extract timing features from flow"""
        # Most timing features already calculated in flow_builder
        features = {}
        
        packets = flow.get("packets", [])
        timestamps = [p["timestamp"] for p in packets]
        
        if len(timestamps) < 2:
            return {}
        
        # Copy from flow if available
        timing_keys = [
            "mean_iat", "std_iat", "min_iat", "max_iat", "median_iat", "burstiness"
        ]
        for key in timing_keys:
            if key in flow:
                features[key] = flow[key]
        
        # Additional timing metrics
        iats = np.diff(timestamps)
        
        # Timing periodicity detection (beaconing)
        features["timing_periodicity"] = TimingFeatures._calc_periodicity(iats)
        
        # IAT distribution moments
        if len(iats) > 1:
            features["iat_skewness"] = float(TimingFeatures._skewness(iats))
            features["iat_kurtosis"] = float(TimingFeatures._kurtosis(iats))
            features["iat_entropy"] = float(TimingFeatures._entropy_continuous(iats))
        
        # Packet arrival rate regularity
        features["packet_arrival_regularity"] = TimingFeatures._calc_arrival_regularity(iats)
        
        # Activity windows (bursty vs sustained)
        features["activity_burstiness"] = TimingFeatures._calc_burstiness(timestamps)
        
        # Beacon detection score
        features["beaconing_score"] = TimingFeatures._calc_beacon_score(iats)
        
        return features
    
    @staticmethod
    def _calc_periodicity(iats: np.ndarray) -> float:
        """
        Detect periodic intervals using FFT
        High values indicate regular beaconing pattern
        """
        if len(iats) < 10:
            return 0.0
        
        try:
            # Compute FFT
            freqs = np.fft.fft(iats)
            power = np.abs(freqs) ** 2
            
            # Find dominant frequency
            if len(power) > 2:
                # Exclude DC component (index 0)
                dominant_power = np.max(power[1:len(power)//2])
                avg_power = np.mean(power[1:len(power)//2])
                
                periodicity = dominant_power / (avg_power + 1e-6)
                return min(periodicity / 100, 1.0)  # Normalize
        except Exception as e:
            logger.debug(f"Periodicity calculation error: {e}")
        
        return 0.0
    
    @staticmethod
    def _calc_arrival_regularity(iats: np.ndarray) -> float:
        """
        Calculate how regular packet arrivals are
        0 = highly irregular, 1 = perfectly regular
        """
        if len(iats) < 2:
            return 0.0
        
        mean_iat = np.mean(iats)
        std_iat = np.std(iats)
        
        if mean_iat == 0:
            return 0.0
        
        # Coefficient of variation (lower = more regular)
        cv = std_iat / mean_iat
        regularity = max(0, 1 - cv)
        
        return min(regularity, 1.0)
    
    @staticmethod
    def _calc_burstiness(timestamps: List[float], window_size: int = 10) -> float:
        """
        Detect bursty traffic patterns
        High burstiness = clustered activity with quiet periods
        """
        if len(timestamps) < window_size * 2:
            return 0.0
        
        # Count packets in non-overlapping windows
        min_ts = min(timestamps)
        max_ts = max(timestamps)
        total_duration = max_ts - min_ts
        
        if total_duration == 0:
            return 0.0
        
        n_windows = max(1, int(len(timestamps) / window_size))
        window_duration = total_duration / n_windows
        
        packet_counts = []
        for i in range(n_windows):
            start = min_ts + i * window_duration
            end = start + window_duration
            count = sum(1 for ts in timestamps if start <= ts < end)
            packet_counts.append(count)
        
        if not packet_counts or np.mean(packet_counts) == 0:
            return 0.0
        
        # Coefficient of variation of packet counts
        burstiness = np.std(packet_counts) / np.mean(packet_counts)
        return min(burstiness, 1.0)
    
    @staticmethod
    def _calc_beacon_score(iats: np.ndarray) -> float:
        """
        Detect beaconing behavior
        Combines regularity + periodicity
        """
        if len(iats) < 5:
            return 0.0
        
        regularity = TimingFeatures._calc_arrival_regularity(iats)
        periodicity = TimingFeatures._calc_periodicity(iats)
        
        # Weighted combination
        beacon_score = regularity * 0.6 + periodicity * 0.4
        
        return min(beacon_score, 1.0)
    
    @staticmethod
    def _skewness(data):
        """Calculate skewness of distribution"""
        if len(data) < 3:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    @staticmethod
    def _kurtosis(data):
        """Calculate kurtosis of distribution"""
        if len(data) < 4:
            return 0.0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    @staticmethod
    def _entropy_continuous(data):
        """Estimate entropy of continuous distribution"""
        if len(data) < 2:
            return 0.0
        
        # Histogram-based entropy
        hist, _ = np.histogram(data, bins=20)
        hist = hist[hist > 0]
        probs = hist / np.sum(hist)
        entropy = -np.sum(probs * np.log2(probs))
        
        return entropy


# Legacy compatibility
def timing_features(flow: Dict) -> Dict:
    """Convenience function for backward compatibility"""
    return TimingFeatures.extract_features(flow)
