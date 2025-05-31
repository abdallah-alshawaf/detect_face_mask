import React from 'react';
import { Shield, Brain } from 'lucide-react';
import './Header.css';

const Header = () => {
    return (
        <header className="header">
            <div className="container">
                <div className="header-content">
                    <div className="logo">
                        <div className="logo-icon">
                            <Shield size={32} />
                            <Brain size={24} className="brain-icon" />
                        </div>
                        <div className="logo-text">
                            <h1>Face Mask Detection</h1>
                            <span>AI-Powered Safety System</span>
                        </div>
                    </div>

                    <nav className="nav">
                        <div className="nav-item">
                            <span className="status-indicator"></span>
                            <span>AI Ready</span>
                        </div>
                    </nav>
                </div>
            </div>
        </header>
    );
};

export default Header; 