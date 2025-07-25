// Simple markdown rendering using marked.js (or fallback to basic)
// You must include marked.js in your HTML for this to work.
// If not available, fallback to basic formatting.

export function renderMarkdown(text) {
    if (window.marked) {
        return marked.parse(text);
    } else {
        // Fallback: basic replacements for bold, italics, code
        return text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
            .replace(/`([^`]+)`/g, '<code>$1</code>')
            .replace(/\n/g, '<br>');
    }
}
