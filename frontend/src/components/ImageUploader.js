import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, Image as ImageIcon, Settings, Zap } from 'lucide-react';
import './ImageUploader.css';

const ImageUploader = ({ onUpload, disabled }) => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [previewUrl, setPreviewUrl] = useState(null);
    const [confidence, setConfidence] = useState(0.5);
    const [showAdvanced, setShowAdvanced] = useState(false);

    const onDrop = useCallback((acceptedFiles) => {
        const file = acceptedFiles[0];
        if (file) {
            setSelectedFile(file);
            const url = URL.createObjectURL(file);
            setPreviewUrl(url);
        }
    }, []);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: {
            'image/*': ['.jpeg', '.jpg', '.png', '.gif', '.bmp', '.webp', '.tiff']
        },
        multiple: false,
        disabled
    });

    const handleUpload = () => {
        if (selectedFile && onUpload) {
            onUpload(selectedFile, confidence);
        }
    };

    const handleReset = () => {
        setSelectedFile(null);
        setPreviewUrl(null);
        if (previewUrl) {
            URL.revokeObjectURL(previewUrl);
        }
    };

    const formatFileSize = (bytes) => {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    };

    return (
        <div className="image-uploader">
            <div className="uploader-header">
                <h2>
                    <ImageIcon size={24} />
                    Upload Image for Analysis
                </h2>
                <p>Upload an image to detect face masks using our AI system</p>
            </div>

            {!selectedFile ? (
                <div
                    {...getRootProps()}
                    className={`dropzone ${isDragActive ? 'active' : ''} ${disabled ? 'disabled' : ''}`}
                >
                    <input {...getInputProps()} />
                    <div className="dropzone-content">
                        <Upload size={48} className="upload-icon" />
                        <h3>
                            {isDragActive ? 'Drop the image here' : 'Drag & drop an image here'}
                        </h3>
                        <p>or click to select a file</p>
                        <div className="supported-formats">
                            <span>Supported formats: JPG, PNG, GIF, BMP, WebP, TIFF</span>
                        </div>
                    </div>
                </div>
            ) : (
                <div className="file-preview">
                    <div className="preview-container">
                        <img src={previewUrl} alt="Preview" className="preview-image" />
                        <div className="file-info">
                            <h4>{selectedFile.name}</h4>
                            <p>Size: {formatFileSize(selectedFile.size)}</p>
                            <p>Type: {selectedFile.type}</p>
                        </div>
                    </div>

                    <div className="preview-actions">
                        <button onClick={handleReset} className="reset-btn" disabled={disabled}>
                            Choose Different Image
                        </button>
                    </div>
                </div>
            )}

            {selectedFile && (
                <div className="upload-controls">
                    <div className="advanced-toggle">
                        <button
                            onClick={() => setShowAdvanced(!showAdvanced)}
                            className="toggle-btn"
                        >
                            <Settings size={16} />
                            Advanced Settings
                        </button>
                    </div>

                    {showAdvanced && (
                        <div className="advanced-settings">
                            <div className="setting-group">
                                <label htmlFor="confidence">
                                    Confidence Threshold: {(confidence * 100).toFixed(0)}%
                                </label>
                                <input
                                    type="range"
                                    id="confidence"
                                    min="0.1"
                                    max="0.9"
                                    step="0.05"
                                    value={confidence}
                                    onChange={(e) => setConfidence(parseFloat(e.target.value))}
                                    className="confidence-slider"
                                />
                                <div className="slider-labels">
                                    <span>More Detections</span>
                                    <span>Higher Accuracy</span>
                                </div>
                            </div>
                        </div>
                    )}

                    <button
                        onClick={handleUpload}
                        disabled={disabled || !selectedFile}
                        className="upload-btn"
                    >
                        <Zap size={20} />
                        Analyze Image
                    </button>
                </div>
            )}

            <div className="upload-info">
                <div className="info-grid">
                    <div className="info-item">
                        <h4>🎯 Detection Classes</h4>
                        <p>With Mask, Without Mask, Incorrect Mask</p>
                    </div>
                    <div className="info-item">
                        <h4>⚡ Processing Speed</h4>
                        <p>Real-time analysis in seconds</p>
                    </div>
                    <div className="info-item">
                        <h4>🔒 Privacy</h4>
                        <p>Images processed locally and securely</p>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ImageUploader; 