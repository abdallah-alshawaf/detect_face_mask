import React, { useState } from 'react';
import { Download, RotateCcw, Eye, BarChart3, CheckCircle, XCircle, AlertTriangle, Shield } from 'lucide-react';
import './ResultsDisplay.css';

const ResultsDisplay = ({ results, onNewUpload }) => {
    const [activeTab, setActiveTab] = useState('comparison');

    const handleDownload = async () => {
        try {
            const response = await fetch(results.download_url);
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = `face_mask_detection_${results.file_id}.png`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        } catch (error) {
            console.error('Download failed:', error);
        }
    };

    const getClassIcon = (className) => {
        switch (className) {
            case 'with_mask':
                return <CheckCircle size={16} className="class-icon success" />;
            case 'without_mask':
                return <XCircle size={16} className="class-icon danger" />;
            case 'mask_weared_incorrect':
                return <AlertTriangle size={16} className="class-icon warning" />;
            default:
                return null;
        }
    };

    const getClassColor = (className) => {
        switch (className) {
            case 'with_mask':
                return '#10b981';
            case 'without_mask':
                return '#ef4444';
            case 'mask_weared_incorrect':
                return '#f59e0b';
            default:
                return '#6b7280';
        }
    };

    const formatClassName = (className) => {
        return className.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    };

    const getSafetyLevelColor = (level) => {
        switch (level) {
            case 'high':
                return '#10b981';
            case 'medium':
                return '#f59e0b';
            case 'low':
                return '#ef4444';
            default:
                return '#6b7280';
        }
    };

    const getSafetyLevelIcon = (level) => {
        switch (level) {
            case 'high':
                return '🛡️';
            case 'medium':
                return '⚠️';
            case 'low':
                return '🚨';
            default:
                return '❓';
        }
    };

    return (
        <div className="results-display">
            <div className="results-header">
                <h2>
                    <Eye size={24} />
                    Detection Results
                </h2>
                <div className="header-actions">
                    <button onClick={handleDownload} className="download-btn">
                        <Download size={16} />
                        Download Result
                    </button>
                    <button onClick={onNewUpload} className="new-upload-btn">
                        <RotateCcw size={16} />
                        New Upload
                    </button>
                </div>
            </div>

            <div className="results-tabs">
                <button
                    className={`tab-btn ${activeTab === 'comparison' ? 'active' : ''}`}
                    onClick={() => setActiveTab('comparison')}
                >
                    <Eye size={16} />
                    Image Comparison
                </button>
                <button
                    className={`tab-btn ${activeTab === 'statistics' ? 'active' : ''}`}
                    onClick={() => setActiveTab('statistics')}
                >
                    <BarChart3 size={16} />
                    Detection Statistics
                </button>
            </div>

            <div className="results-content">
                {activeTab === 'comparison' && (
                    <div className="image-comparison">
                        <div className="image-container">
                            <div className="image-wrapper">
                                <h3>Original Image</h3>
                                <img
                                    src={results.original_image}
                                    alt="Original"
                                    className="result-image"
                                />
                            </div>
                            <div className="comparison-arrow">
                                →
                            </div>
                            <div className="image-wrapper">
                                <h3>AI Detection Result</h3>
                                <img
                                    src={results.processed_image}
                                    alt="Processed"
                                    className="result-image"
                                />

                                {/* Mask Detection Summary */}
                                {results.analysis && (
                                    <div className="mask-detection-summary">
                                        <div className={`safety-badge ${results.analysis.safety_level}`}>
                                            <span className="safety-icon">
                                                {getSafetyLevelIcon(results.analysis.safety_level)}
                                            </span>
                                            <span className="safety-text">
                                                {results.analysis.summary}
                                            </span>
                                        </div>

                                        {results.analysis.details && results.analysis.details.total_faces > 0 && (
                                            <div className="compliance-details">
                                                <div className="compliance-rate">
                                                    <Shield size={16} />
                                                    <span>Compliance Rate: {results.analysis.compliance_rate}%</span>
                                                </div>
                                                <div className="face-breakdown">
                                                    {results.analysis.details.with_mask > 0 && (
                                                        <span className="face-count with-mask">
                                                            ✅ {results.analysis.details.with_mask} with mask
                                                        </span>
                                                    )}
                                                    {results.analysis.details.without_mask > 0 && (
                                                        <span className="face-count without-mask">
                                                            ❌ {results.analysis.details.without_mask} without mask
                                                        </span>
                                                    )}
                                                    {results.analysis.details.incorrect_mask > 0 && (
                                                        <span className="face-count incorrect-mask">
                                                            ⚠️ {results.analysis.details.incorrect_mask} incorrect mask
                                                        </span>
                                                    )}
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>
                        </div>

                        <div className="detection-summary">
                            <h4>Detection Summary</h4>
                            <div className="summary-grid">
                                <div className="summary-item">
                                    <span className="summary-label">Total Detections:</span>
                                    <span className="summary-value">{results.statistics.total_detections}</span>
                                </div>
                                <div className="summary-item">
                                    <span className="summary-label">Confidence Threshold:</span>
                                    <span className="summary-value">{(results.confidence_threshold * 100).toFixed(0)}%</span>
                                </div>
                                <div className="summary-item">
                                    <span className="summary-label">Processing Time:</span>
                                    <span className="summary-value">~2-3 seconds</span>
                                </div>
                                {results.demo_mode && (
                                    <div className="summary-item demo-mode">
                                        <span className="summary-label">Mode:</span>
                                        <span className="summary-value">Demo Mode</span>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                )}

                {activeTab === 'statistics' && (
                    <div className="statistics-panel">
                        <div className="stats-overview">
                            <h3>Detection Overview</h3>
                            <div className="stats-cards">
                                <div className="stat-card total">
                                    <div className="stat-icon">🎯</div>
                                    <div className="stat-content">
                                        <h4>Total Detections</h4>
                                        <p className="stat-number">{results.statistics.total_detections}</p>
                                    </div>
                                </div>
                                <div className="stat-card confidence">
                                    <div className="stat-icon">📊</div>
                                    <div className="stat-content">
                                        <h4>Confidence Level</h4>
                                        <p className="stat-number">{(results.confidence_threshold * 100).toFixed(0)}%</p>
                                    </div>
                                </div>
                                {results.analysis && (
                                    <div className="stat-card safety">
                                        <div className="stat-icon">{getSafetyLevelIcon(results.analysis.safety_level)}</div>
                                        <div className="stat-content">
                                            <h4>Safety Level</h4>
                                            <p className="stat-number" style={{ color: getSafetyLevelColor(results.analysis.safety_level) }}>
                                                {results.analysis.safety_level.toUpperCase()}
                                            </p>
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>

                        <div className="class-breakdown">
                            <h3>Detection Breakdown by Class</h3>
                            <div className="class-list">
                                {Object.entries(results.statistics.class_counts).map(([className, count]) => (
                                    <div key={className} className="class-item">
                                        <div className="class-header">
                                            {getClassIcon(className)}
                                            <span className="class-name">{formatClassName(className)}</span>
                                            <span className="class-count">{count} detected</span>
                                        </div>
                                        <div className="class-bar">
                                            <div
                                                className="class-fill"
                                                style={{
                                                    width: `${(count / results.statistics.total_detections) * 100}%`,
                                                    backgroundColor: getClassColor(className)
                                                }}
                                            ></div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {results.statistics.detections && results.statistics.detections.length > 0 && (
                            <div className="detailed-detections">
                                <h3>Detailed Detection Results</h3>
                                <div className="detections-table">
                                    <div className="table-header">
                                        <span>Class</span>
                                        <span>Confidence</span>
                                        <span>Bounding Box</span>
                                    </div>
                                    {results.statistics.detections.map((detection, index) => (
                                        <div key={index} className="table-row">
                                            <div className="detection-class">
                                                {getClassIcon(detection.class)}
                                                {formatClassName(detection.class)}
                                            </div>
                                            <div className="detection-confidence">
                                                {(detection.confidence * 100).toFixed(1)}%
                                            </div>
                                            <div className="detection-bbox">
                                                ({detection.bbox.join(', ')})
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>
                )}
            </div>

            <div className="results-footer">
                <div className="footer-info">
                    <p>
                        <strong>AI Model:</strong> YOLOv8 Face Mask Detection
                        {results.demo_mode && <span className="demo-badge"> (Demo Mode)</span>}
                    </p>
                    <p>
                        <strong>Classes:</strong> With Mask (Green), Without Mask (Red), Incorrect Mask (Orange)
                    </p>
                </div>
            </div>
        </div>
    );
};

export default ResultsDisplay; 