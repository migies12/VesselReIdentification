import { useNavigate } from "react-router-dom";
import "../styles/Home.css";

export default function Home() {
  const navigate = useNavigate();

  return (
    <div className="app">
      <header className="app-header">
        <h1>Dark Vessel Re-Identification</h1>
      </header>
      
      <main className="main-panel menu-container">
        <button 
          className="nav-controls button btn-reid btn-large" 
          onClick={() => navigate("/demo")}
        >
          Dark Vessel Identification
        </button>
        
        <button 
          className="nav-controls button btn-reid btn-large" 
          onClick={() => navigate("/events")}
        >
          Browse Recent Events
        </button>
      </main>
    </div>
  );
}