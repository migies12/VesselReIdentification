import { Routes, Route } from "react-router-dom";
import Admin from "./pages/Admin";
import Dashboard from "./pages/Dashboard";
import Home from "./pages/Home";
import { fetchDemoEvents, fetchEvents } from "./utils/api";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/admin" element={<Admin />} />
      <Route path="/demo" element={
        <Dashboard
          fetchEvents={fetchDemoEvents}
        />
      } />
      <Route path="/events" element={
        <Dashboard 
          fetchEvents={fetchEvents}
        />
      } />
      <Route path="*" element={<div>404 - Page Not Found</div>} />
    </Routes>
  );
}