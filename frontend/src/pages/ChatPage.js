import React, { useState, useEffect, useRef } from 'react';
import { 
  Box, 
  Paper, 
  Typography, 
  Button, 
  CircularProgress, 
  Snackbar, 
  Alert,
  Card,
  CardContent,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Chip
} from '@mui/material';
import DownloadIcon from '@mui/icons-material/Download';
import RefreshIcon from '@mui/icons-material/Refresh';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import InfoIcon from '@mui/icons-material/Info';
import BugReportIcon from '@mui/icons-material/BugReport';
import WarningIcon from '@mui/icons-material/Warning';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ChatMessage from '../components/ChatMessage';
import ChatInput from '../components/ChatInput';
import SettingsDialog from '../components/SettingsDialog';
import { v4 as uuidv4 } from 'uuid';
import { chatService } from '../services/chatService';

const ChatPage = () => {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState('');
  const [error, setError] = useState(null);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [settings, setSettings] = useState({
    webSearchEnabled: true,
    creativity: 7,
    darkMode: false,
    showTimestamps: true,
  });
  const [backendAvailable, setBackendAvailable] = useState(true);
  const [retryCount, setRetryCount] = useState(0);
  const [healthStatus, setHealthStatus] = useState(null);
  const [showDiagnostics, setShowDiagnostics] = useState(false);
  
  const messagesEndRef = useRef(null);

  // Get a user-friendly error message based on the current health status
  const getErrorMessage = (status) => {
    if (!status) return 'Unable to connect to the AI service.';
    
    if (status.connectionError) {
      return `Connection to the backend failed: ${status.connectionError}`;
    }
    
    if (!status.modelExists) {
      return 'The AI model file is missing on the server. Please contact support.';
    }
    
    if (!status.modelLoaded && status.errorMessage) {
      return `The AI model couldn't be loaded: ${status.errorMessage}`;
    }
    
    if (!status.isHealthy) {
      return 'The AI service is experiencing issues. Our team has been notified.';
    }
    
    return 'Unknown error with the AI service.';
  };

  // Initialize chat session
  useEffect(() => {
    const initSession = async () => {
      try {
        setLoading(true);
        
        // Check if backend is available and get detailed health status
        const status = await chatService.getHealthStatus();
        setHealthStatus(status);
        setBackendAvailable(status.isHealthy && status.modelLoaded);
        
        if (!status.isHealthy || !status.modelLoaded) {
          const errorMsg = getErrorMessage(status);
          setError(errorMsg);
          console.warn('Backend service issue:', errorMsg);
          setShowDiagnostics(true);
        }
        
        // Create a new chat session
        const session = await chatService.createSession();
        setSessionId(session.sessionId);
        
        // Add welcome message
        let welcomeMessage = {
          role: 'assistant',
          content: 'Hello! I\'m Backdoor AI. How can I help you today?',
          intent: 'greeting',
          timestamp: new Date().toISOString(),
        };
        
        // Modify welcome message if there are issues
        if (!status?.isHealthy || !status?.modelLoaded) {
          welcomeMessage = {
            role: 'assistant',
            content: 'Hello! I\'m Backdoor AI. I\'m having some technical difficulties at the moment. ' +
                    'You can try sending a message, but I might not be able to respond properly.',
            intent: 'system_warning',
            timestamp: new Date().toISOString(),
          };
        }
        
        setMessages([welcomeMessage]);
      } catch (err) {
        console.error('Failed to initialize chat session:', err);
        setError('Failed to start chat session. Please try again.');
        setShowDiagnostics(true);
      } finally {
        setLoading(false);
      }
    };
    
    initSession();
    
    // Set up periodic health checks
    const healthCheckInterval = setInterval(async () => {
      try {
        const status = await chatService.checkHealth();
        const newStatus = await chatService.getHealthStatus();
        setHealthStatus(newStatus);
        setBackendAvailable(newStatus.isHealthy && newStatus.modelLoaded);
        
        // If state changed from unavailable to available, notify user
        if ((!backendAvailable) && (newStatus.isHealthy && newStatus.modelLoaded)) {
          setMessages((prevMessages) => [
            ...prevMessages,
            {
              role: 'assistant',
              content: 'Connection restored! The AI service is now available.',
              intent: 'system',
              timestamp: new Date().toISOString(),
            },
          ]);
        }
      } catch (error) {
        console.error('Health check failed:', error);
      }
    }, 30000); // Check every 30 seconds
    
    return () => clearInterval(healthCheckInterval);
  }, []);

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Toggle diagnostic panel
  const toggleDiagnostics = () => {
    setShowDiagnostics(!showDiagnostics);
  };

  const handleSendMessage = async (content, useWebSearch = settings.webSearchEnabled) => {
    if (!content.trim()) return;
    
    // Add user message to chat
    const userMessage = {
      role: 'user',
      content,
      timestamp: new Date().toISOString(),
    };
    
    setMessages((prevMessages) => [...prevMessages, userMessage]);
    setLoading(true);
    setError(null);
    
    try {
      // Re-check health status before sending
      const currentStatus = await chatService.getHealthStatus();
      setHealthStatus(currentStatus);
      
      if (!currentStatus.isHealthy || !currentStatus.modelLoaded) {
        throw new Error(getErrorMessage(currentStatus));
      }
      
      // Call API to get AI response with retry mechanism
      const response = await chatService.retryRequest(
        () => chatService.sendMessage(sessionId, userMessage, useWebSearch),
        3, // max retries
        1000 // delay between retries in ms
      );
      
      // Add AI response to chat
      setMessages((prevMessages) => [...prevMessages, response]);
      // Reset retry count on success
      setRetryCount(0);
    } catch (err) {
      console.error('Error sending message:', err);
      const errorMessage = err.message || 'Failed to get a response. Please try again.';
      setError(errorMessage);
      
      // Increment retry count
      setRetryCount(prev => prev + 1);
      
      // Add error message
      setMessages((prevMessages) => [
        ...prevMessages,
        {
          role: 'assistant',
          content: 'Sorry, I encountered an error processing your request. ' + 
                   (healthStatus && !healthStatus.modelLoaded 
                    ? 'The AI model is not currently loaded on the server.' 
                    : 'Please try again.'),
          intent: 'error',
          timestamp: new Date().toISOString(),
        },
      ]);
      
      // If we've had multiple failures, suggest troubleshooting
      if (retryCount >= 2) {
        setError('Multiple errors detected. Click "View Diagnostics" to see technical details.');
        setShowDiagnostics(true);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleExportChat = () => {
    // Create a JSON file with the chat history
    const chatData = {
      sessionId,
      messages,
      exportedAt: new Date().toISOString(),
      systemInfo: {
        healthStatus,
        retryCount,
        userAgent: navigator.userAgent
      }
    };
    
    const blob = new Blob([JSON.stringify(chatData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `backdoor-ai-chat-${new Date().toLocaleDateString().replace(/\//g, '-')}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleCloseError = () => {
    setError(null);
  };

  const handleRetryConnection = async () => {
    setLoading(true);
    setError(null);
    
    try {
      // Check backend health with fresh request
      const status = await chatService.getHealthStatus();
      setHealthStatus(status);
      setBackendAvailable(status.isHealthy && status.modelLoaded);
      
      if (status.isHealthy && status.modelLoaded) {
        setMessages((prevMessages) => [
          ...prevMessages,
          {
            role: 'assistant',
            content: 'Connection restored! You can continue chatting.',
            intent: 'system',
            timestamp: new Date().toISOString(),
          },
        ]);
      } else {
        setError(getErrorMessage(status));
        setShowDiagnostics(true);
      }
    } catch (err) {
      console.error('Error checking backend health:', err);
      setError('Failed to check backend connection. Please try again.');
      setShowDiagnostics(true);
    } finally {
      setLoading(false);
    }
  };
  
  // Render the diagnostic panel with system status
  const renderDiagnosticPanel = () => {
    return (
      <Card sx={{ mb: 3, bgcolor: '#f8f9fa', borderLeft: '4px solid #ff9800' }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
            <BugReportIcon color="warning" sx={{ mr: 1 }} />
            <Typography variant="h6">System Diagnostics</Typography>
          </Box>
          
          <Divider sx={{ mb: 2 }} />
          
          <List dense>
            <ListItem>
              <ListItemIcon>
                <Chip 
                  label={healthStatus?.isHealthy ? "Healthy" : "Degraded"} 
                  color={healthStatus?.isHealthy ? "success" : "error"}
                  size="small"
                />
              </ListItemIcon>
              <ListItemText 
                primary="Backend Status" 
                secondary={`Last checked: ${healthStatus?.timestamp || 'Unknown'}`} 
              />
            </ListItem>
            
            <ListItem>
              <ListItemIcon>
                <Chip 
                  label={healthStatus?.modelLoaded ? "Loaded" : "Not Loaded"} 
                  color={healthStatus?.modelLoaded ? "success" : "error"}
                  size="small"
                />
              </ListItemIcon>
              <ListItemText 
                primary="AI Model Status" 
                secondary={healthStatus?.modelExists ? "Model file exists" : "Model file missing"} 
              />
            </ListItem>
            
            {healthStatus?.errorMessage && (
              <ListItem>
                <ListItemIcon>
                  <ErrorOutlineIcon color="error" />
                </ListItemIcon>
                <ListItemText 
                  primary="Error Details" 
                  secondary={healthStatus.errorMessage} 
                />
              </ListItem>
            )}
          </List>
          
          {healthStatus?.modelDetails && (
            <Accordion sx={{ mt: 2 }}>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Typography><InfoIcon sx={{ mr: 1, fontSize: '0.9rem', verticalAlign: 'middle' }} /> Model Details</Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Typography variant="body2">Description: {healthStatus.modelDetails.description || 'None'}</Typography>
                <Typography variant="body2">Author: {healthStatus.modelDetails.author || 'Unknown'}</Typography>
                <Typography variant="body2">Load Time: {healthStatus.modelDetails.load_time_sec?.toFixed(2) || 'N/A'} seconds</Typography>
                
                {healthStatus.modelDetails.inputs && (
                  <Box mt={1}>
                    <Typography variant="body2" fontWeight="bold">Inputs:</Typography>
                    <ul style={{ margin: 0, paddingLeft: 20 }}>
                      {healthStatus.modelDetails.inputs.map((input, idx) => (
                        <li key={idx}><Typography variant="body2">{input}</Typography></li>
                      ))}
                    </ul>
                  </Box>
                )}
                
                {healthStatus.modelDetails.outputs && (
                  <Box mt={1}>
                    <Typography variant="body2" fontWeight="bold">Outputs:</Typography>
                    <ul style={{ margin: 0, paddingLeft: 20 }}>
                      {healthStatus.modelDetails.outputs.map((output, idx) => (
                        <li key={idx}><Typography variant="body2">{output}</Typography></li>
                      ))}
                    </ul>
                  </Box>
                )}
              </AccordionDetails>
            </Accordion>
          )}
          
          <Box mt={2} display="flex" justifyContent="flex-end">
            <Button 
              variant="outlined" 
              color="primary" 
              startIcon={<RefreshIcon />}
              onClick={handleRetryConnection}
              sx={{ mr: 1 }}
              disabled={loading}
            >
              Retry Connection
            </Button>
            <Button 
              variant="outlined" 
              onClick={() => setShowDiagnostics(false)}
              disabled={loading}
            >
              Hide Diagnostics
            </Button>
          </Box>
        </CardContent>
      </Card>
    );
  };

  return (
    <Box sx={{ height: 'calc(100vh - 120px)', display: 'flex', flexDirection: 'column' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
        <Box>
          <Button
            variant="outlined"
            color="primary"
            onClick={handleRetryConnection}
            disabled={loading || (backendAvailable && !showDiagnostics)}
            sx={{ mr: 1 }}
          >
            Retry Connection
          </Button>
          
          <Button
            variant="outlined"
            color="secondary"
            startIcon={showDiagnostics ? null : <BugReportIcon />}
            onClick={toggleDiagnostics}
            disabled={loading}
          >
            {showDiagnostics ? 'Hide Diagnostics' : 'View Diagnostics'}
          </Button>
        </Box>
        
        <Button
          variant="outlined"
          startIcon={<DownloadIcon />}
          onClick={handleExportChat}
          disabled={messages.length <= 1}
        >
          Export Chat
        </Button>
      </Box>
      
      {/* Display system status banner when there are issues */}
      {!backendAvailable && !showDiagnostics && (
        <Alert 
          severity="warning" 
          sx={{ mb: 2 }}
          action={
            <Button 
              color="inherit" 
              size="small" 
              onClick={toggleDiagnostics}
            >
              View Details
            </Button>
          }
        >
          {getErrorMessage(healthStatus)}
        </Alert>
      )}
      
      {/* Diagnostic panel */}
      {showDiagnostics && renderDiagnosticPanel()}
      
      <Paper
        elevation={3}
        sx={{
          flex: 1,
          p: 3,
          borderRadius: 2,
          overflow: 'hidden',
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        <Box sx={{ flex: 1, overflowY: 'auto', mb: 2, px: 1 }}>
          {messages.length === 0 ? (
            <Box
              sx={{
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'center',
                alignItems: 'center',
                color: 'text.secondary',
              }}
            >
              <Typography variant="h5" gutterBottom>
                Welcome to Backdoor AI
              </Typography>
              <Typography variant="body1">
                Start a conversation by typing a message below.
              </Typography>
              
              {!backendAvailable && (
                <Alert severity="warning" sx={{ mt: 2, width: '100%', maxWidth: '500px' }}>
                  The AI model is currently not available. You can still try sending messages,
                  but responses may be limited.
                </Alert>
              )}
            </Box>
          ) : (
            messages.map((msg, index) => (
              <ChatMessage key={index} message={msg} showTimestamp={settings.showTimestamps} />
            ))
          )}
          {loading && (
            <Box sx={{ display: 'flex', justifyContent: 'center', my: 2 }}>
              <CircularProgress size={24} />
            </Box>
          )}
          <div ref={messagesEndRef} />
        </Box>
        
        <ChatInput
          onSendMessage={handleSendMessage}
          onToggleSettings={() => setSettingsOpen(true)}
          disabled={loading || (!backendAvailable && retryCount >= 2)}
          webSearchEnabled={settings.webSearchEnabled}
        />
        
        {!backendAvailable && (
          <Alert severity="info" sx={{ mt: 2, mb: 0 }}>
            {retryCount >= 2 
              ? "The AI service is currently unavailable. Please try again later." 
              : "The AI service may have limited functionality. You can try sending a message, but it might not work properly."}
          </Alert>
        )}
      </Paper>
      
      <SettingsDialog
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        settings={settings}
        onSettingsChange={setSettings}
      />
      
      <Snackbar 
        open={!!error} 
        autoHideDuration={6000} 
        onClose={handleCloseError}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert 
          onClose={handleCloseError} 
          severity="error" 
          sx={{ width: '100%' }}
          action={
            !showDiagnostics ? (
              <Button color="inherit" size="small" onClick={toggleDiagnostics}>
                Diagnostics
              </Button>
            ) : undefined
          }
        >
          {error}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default ChatPage;
