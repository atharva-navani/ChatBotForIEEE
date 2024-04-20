// Example of a script tag endpoint using Node.js and Express
const express = require('express');
const app = express();

// Endpoint to generate the script tag
app.get('/script-tag', (req, res) => {
    const scriptCode = `
        <!-- Chatbot script -->
        <script src="https://your-chatbot-server.com/chatbot-script.js"></script>
        <!-- End of Chatbot script -->
    `;
    res.send(scriptCode);
});

// Start the server
const port = process.env.PORT || 3000;
app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});


