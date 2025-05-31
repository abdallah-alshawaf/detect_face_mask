import React from 'react';
import { Heart, Github, Shield } from 'lucide-react';
import './Footer.css';

const Footer = () => {
    return (
        <footer className="footer">
            <div className="container">
                <div className="footer-content">
                    <div className="footer-section">
                        <div className="footer-logo">
                            <Shield size={24} />
                            <span>Face Mask Detection</span>
                        </div>
                        <p>AI-powered safety system for real-time face mask detection</p>
                    </div>

                    <div className="footer-section">
                        <h4>Technology</h4>
                        <ul>
                            <li>YOLOv8 Deep Learning</li>
                            <li>React Frontend</li>
                            <li>Flask Backend</li>
                            <li>Computer Vision</li>
                        </ul>
                    </div>

                    <div className="footer-section">
                        <h4>Features</h4>
                        <ul>
                            <li>Real-time Detection</li>
                            <li>High Accuracy</li>
                            <li>Multiple Classes</li>
                            <li>Easy to Use</li>
                        </ul>
                    </div>

                    <div className="footer-section">
                        <h4>Safety Classes</h4>
                        <ul>
                            <li><span className="class-indicator with-mask"></span>With Mask</li>
                            <li><span className="class-indicator without-mask"></span>Without Mask</li>
                            <li><span className="class-indicator incorrect-mask"></span>Incorrect Mask</li>
                        </ul>
                    </div>
                </div>

                <div className="footer-bottom">
                    <div className="footer-credits">
                        <p>
                            Made with <Heart size={16} className="heart-icon" /> for public safety
                        </p>
                    </div>
                    <div className="footer-links">
                        <a href="#" className="footer-link">
                            <Github size={16} />
                            Source Code
                        </a>
                    </div>
                </div>
            </div>
        </footer>
    );
};

export default Footer; 