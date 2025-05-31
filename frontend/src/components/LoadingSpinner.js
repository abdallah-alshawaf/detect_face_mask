import React from 'react';
import './LoadingSpinner.css';

const LoadingSpinner = ({ size = 'medium', text = '' }) => {
    return (
        <div className={`loading-spinner ${size}`}>
            <div className="spinner-container">
                <div className="spinner">
                    <div className="spinner-ring"></div>
                    <div className="spinner-ring"></div>
                    <div className="spinner-ring"></div>
                    <div className="spinner-ring"></div>
                </div>
                <div className="ai-brain">
                    🧠
                </div>
            </div>
            {text && <p className="loading-text">{text}</p>}
        </div>
    );
};

export default LoadingSpinner; 