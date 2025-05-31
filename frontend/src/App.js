import React, { useState, useEffect } from 'react';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import ImageUploader from './components/ImageUploader';
import ResultsDisplay from './components/ResultsDisplay';
import Header from './components/Header';
import Footer from './components/Footer';
import LoadingSpinner from './components/LoadingSpinner';
import './App.css';

function App() {
    const [isProcessing, setIsProcessing] = useState(false);
    const [results, setResults] = useState(null);
    const [apiStatus, setApiStatus] = useState('checking');

    // Check API health on component mount
    useEffect(() => {
        checkApiHealth();
    }, []);

    const checkApiHealth = async () => {
        try {
            const response = await fetch('/api/health');
            const data = await response.json();

            if (data.status === 'healthy' && data.detector_ready) {
                setApiStatus('ready');
                toast.success('AI Detection System Ready!', {
                    position: "top-right",
                    autoClose: 3000,
                });
            } else {
                setApiStatus('error');
                toast.error('Detection system not ready. Please check the backend.', {
                    position: "top-right",
                    autoClose: 5000,
                });
            }
        } catch (error) {
            setApiStatus('error');
            toast.error('Cannot connect to detection service. Please start the backend.', {
                position: "top-right",
                autoClose: 5000,
            });
        }
    };

    const handleImageUpload = async (file, confidence) => {
        setIsProcessing(true);
        setResults(null);

        const formData = new FormData();
        formData.append('image', file);
        formData.append('confidence', confidence);

        try {
            const response = await fetch('/api/detect', {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();

            if (data.success) {
                setResults(data);
                toast.success('Image processed successfully!', {
                    position: "top-right",
                    autoClose: 3000,
                });
            } else {
                throw new Error(data.error || 'Processing failed');
            }
        } catch (error) {
            toast.error(`Error: ${error.message}`, {
                position: "top-right",
                autoClose: 5000,
            });
            console.error('Upload error:', error);
        } finally {
            setIsProcessing(false);
        }
    };

    const handleNewUpload = () => {
        setResults(null);
    };

    return (
        <div className="App">
            <Header />

            <main className="main-content">
                <div className="container">
                    {apiStatus === 'checking' && (
                        <div className="api-status checking">
                            <LoadingSpinner />
                            <p>Connecting to AI Detection System...</p>
                        </div>
                    )}

                    {apiStatus === 'error' && (
                        <div className="api-status error">
                            <div className="error-icon">⚠️</div>
                            <h3>Service Unavailable</h3>
                            <p>The face mask detection service is not available.</p>
                            <button onClick={checkApiHealth} className="retry-btn">
                                Retry Connection
                            </button>
                        </div>
                    )}

                    {apiStatus === 'ready' && (
                        <>
                            {!results && !isProcessing && (
                                <div className="upload-section fade-in">
                                    <ImageUploader
                                        onUpload={handleImageUpload}
                                        disabled={isProcessing}
                                    />
                                </div>
                            )}

                            {isProcessing && (
                                <div className="processing-section fade-in">
                                    <LoadingSpinner />
                                    <h3>Processing Image...</h3>
                                    <p>Our AI is analyzing the image for face mask detection</p>
                                </div>
                            )}

                            {results && !isProcessing && (
                                <div className="results-section fade-in">
                                    <ResultsDisplay
                                        results={results}
                                        onNewUpload={handleNewUpload}
                                    />
                                </div>
                            )}
                        </>
                    )}
                </div>
            </main>

            <Footer />

            <ToastContainer
                position="top-right"
                autoClose={3000}
                hideProgressBar={false}
                newestOnTop={false}
                closeOnClick
                rtl={false}
                pauseOnFocusLoss
                draggable
                pauseOnHover
                theme="light"
            />
        </div>
    );
}

export default App; 