import { useState, useRef } from 'react';
import { UploadCloud, CheckCircle, XCircle, X } from 'lucide-react';

function App() {
  const [refFile, setRefFile] = useState(null);
  const [verFile, setVerFile] = useState(null);
  const [refPreview, setRefPreview] = useState(null);
  const [verPreview, setVerFilePreview] = useState(null);
  
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleDrag = (e) => { e.preventDefault(); e.stopPropagation(); };

  const handleDrop = (e, type) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0], type);
    }
  };

  const handleChange = (e, type) => {
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0], type);
    }
  };

  const processFile = (file, type) => {
    setError(null);
    setResult(null);
    if (file.type.startsWith('image/')) {
      const url = URL.createObjectURL(file);
      if (type === 'ref') { setRefFile(file); setRefPreview(url); }
      else { setVerFile(file); setVerFilePreview(url); }
    } else if (file.type === 'application/pdf') {
      // Avoid browser preview for PDFs, use an icon instead
      if (type === 'ref') { setRefFile(file); setRefPreview('pdf'); }
      else { setVerFile(file); setVerFilePreview('pdf'); }
    } else {
      setError("Please upload an Image or PDF format.");
    }
  };

  const clearSlot = (type) => {
    setResult(null);
    if (type === 'ref') { setRefFile(null); setRefPreview(null); }
    else { setVerFile(null); setVerFilePreview(null); }
  };

  const triggerVerify = async () => {
    if (!refFile || !verFile) return;
    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('reference', refFile);
    formData.append('verification', verFile);

    try {
      // Backend is Flask running on port 5000
      const res = await fetch('http://localhost:5000/api/verify-signature', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (!res.ok || !data.success) throw new Error(data.error || "Verification failed");
      
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const renderSlot = (title, type, file, preview, setHover, hover) => (
    <div 
      className={`upload-zone ${hover ? 'drag-active' : ''}`}
      onDragEnter={(e) => {handleDrag(e); setHover(true)}}
      onDragLeave={(e) => {handleDrag(e); setHover(false)}}
      onDragOver={handleDrag}
      onDrop={(e) => { setHover(false); handleDrop(e, type); }}
      onClick={() => document.getElementById(`fileInput-${type}`).click()}
    >
      <input 
        id={`fileInput-${type}`} 
        type="file" 
        accept="image/*,application/pdf" 
        style={{ display: 'none' }} 
        onChange={(e) => handleChange(e, type)} 
      />
      
      {preview ? (
        <>
          {preview === 'pdf' ? (
             <div style={{color:'white', marginTop:'40px'}}>
               <CheckCircle size={48} style={{color:'var(--success)', marginBottom:'10px'}}/>
               <h3>PDF Document Uploaded</h3>
               <p style={{color:'var(--text-secondary)'}}>{file?.name}</p>
             </div>
          ) : (
            <img src={preview} alt="preview" className="file-preview" />
          )}
          <button 
            className="clear-btn" 
            onClick={(e) => { e.stopPropagation(); clearSlot(type); }}
          >
            <X size={16} />
          </button>
        </>
      ) : (
        <div style={{ pointerEvents: 'none' }}>
          <UploadCloud className="upload-icon" size={48} />
          <h3 style={{marginBottom:'10px', color:'white'}}>{title}</h3>
          <p style={{color:'var(--text-secondary)', fontSize:'0.9rem'}}>Drag & drop or click to upload</p>
          <p style={{color:'var(--text-secondary)', fontSize:'0.8rem', marginTop:'5px'}}>Supports Image and PDF</p>
        </div>
      )}
    </div>
  );

  const [refHover, setRefHover] = useState(false);
  const [verHover, setVerHover] = useState(false);

  return (
    <main className="glass-panel">
      <h1>Signature Verification</h1>
      <p className="subtitle">Securely match authentication signatures using advanced structural logic</p>

      {error && (
        <div style={{background:'rgba(239, 68, 68, 0.2)', color:'#fca5a5', padding:'15px', borderRadius:'10px', marginBottom:'20px', textAlign:'center', border:'1px solid #ef4444'}}>
          {error}
        </div>
      )}

      <div className="upload-grid">
        {renderSlot("Reference Signature", 'ref', refFile, refPreview, setRefHover, refHover)}
        {renderSlot("Test Signature", 'ver', verFile, verPreview, setVerHover, verHover)}
      </div>

      <button 
        className="verify-btn" 
        onClick={triggerVerify} 
        disabled={!refFile || !verFile || loading}
      >
        {loading ? <div className="loader"></div> : "Verify Signatures"}
      </button>

      {result && (
        <div className="results-container">
          <div 
            className="score-circle" 
            style={{
              color: result.color === 'green' ? 'var(--success)' : 'var(--error)',
            }}
          >
            {result.match_score}%
          </div>
          <h2 
            className="status-text"
            style={{ color: result.color === 'green' ? 'var(--success)' : 'var(--error)' }}
          >
            {result.status}
          </h2>
          <p style={{ color: 'var(--text-secondary)', marginTop: '10px' }}>
            {result.color === 'green' ? 'Signatures are an acceptable structural match.' : 'Signatures present significant structural derivations.'}
          </p>
        </div>
      )}
    </main>
  );
}

export default App;
