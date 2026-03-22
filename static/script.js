const conversationHistory = [];
const skillTags = ['Python', 'Excel', 'Communication'];
const interestTags = ['Technology', 'Finance'];
let isDark = true;

function toggleTheme() {
  isDark = !isDark;
  document.documentElement.setAttribute('data-theme', isDark ? 'dark' : 'light');
  document.getElementById('theme-btn').textContent = isDark ? '🌙' : '☀️';
}

function openSidebar() {
  document.getElementById('sidebar').classList.add('open');
  document.getElementById('overlay').classList.add('active');
}

function closeSidebar() {
  document.getElementById('sidebar').classList.remove('open');
  document.getElementById('overlay').classList.remove('active');
}

function renderTags() {
  const colors = ['tag-accent', 'tag-teal', 'tag-coral'];
  document.getElementById('skill-tags').innerHTML =
    skillTags.map((t, i) => `<span class="tag ${colors[i % 3]}" onclick="removeTag('skill','${t}')">${t} x</span>`).join('');
  document.getElementById('interest-tags').innerHTML =
    interestTags.map((t, i) => `<span class="tag ${colors[(i+1) % 3]}" onclick="removeTag('interest','${t}')">${t} x</span>`).join('');
}

function addTag(type) {
  const input = document.getElementById(type + '-input');
  const val = input.value.trim();
  if (!val) return;
  const arr = type === 'skill' ? skillTags : interestTags;
  if (!arr.includes(val)) arr.push(val);
  input.value = '';
  renderTags();
}

function removeTag(type, val) {
  const arr = type === 'skill' ? skillTags : interestTags;
  const i = arr.indexOf(val);
  if (i > -1) arr.splice(i, 1);
  renderTags();
}

document.getElementById('skill-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') { e.preventDefault(); addTag('skill'); }
});
document.getElementById('interest-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') { e.preventDefault(); addTag('interest'); }
});

function getProfile() {
  return {
    stream: document.getElementById('stream').value,
    grade: document.getElementById('grade').value,
    workexp: document.getElementById('workexp').value,
    skills: [...skillTags],
    interests: [...interestTags]
  };
}

function renderMarkdown(raw) {
  const lines = raw.split('\n');
  let html = '';
  let inUl = false;
  let inOl = false;

  function closeUl() { if (inUl) { html += '</ul>'; inUl = false; } }
  function closeOl() { if (inOl) { html += '</ol>'; inOl = false; } }
  function closeLists() { closeUl(); closeOl(); }

  function inlineFormat(line) {
    return line
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>')
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.+?)\*/g, '<em>$1</em>')
      .replace(/__(.+?)__/g, '<strong>$1</strong>')
      .replace(/_(.+?)_/g, '<em>$1</em>')
      .replace(/`(.+?)`/g, '<code>$1</code>');
  }

  for (let i = 0; i < lines.length; i++) {
    const raw_line = lines[i];
    const trimmed = raw_line.trim();
    const line = inlineFormat(raw_line);
    const trimmedFormatted = inlineFormat(trimmed);

    // skip empty lines
    if (trimmed === '') {
      closeLists();
      continue;
    }

    // horizontal rule
    if (/^[-*]{3,}$/.test(trimmed)) {
      closeLists();
      html += '<hr>';
      continue;
    }

    // headings: ##### #### ### ## #
    if (/^#{1,6}\s/.test(trimmed)) {
      closeLists();
      const level = trimmed.match(/^(#{1,6})\s/)[1].length;
      const content = inlineFormat(trimmed.replace(/^#{1,6}\s+/, ''));
      const tag = level <= 2 ? 'h2' : 'h3';
      html += '<' + tag + '>' + content + '</' + tag + '>';
      continue;
    }

    // unordered list: - * + • ➤ ▸ → (with any leading spaces)
    if (/^\s*[-*+•➤▸→]\s+/.test(raw_line)) {
      closeOl();
      if (!inUl) { html += '<ul>'; inUl = true; }
      const content = inlineFormat(raw_line.replace(/^\s*[-*+•➤▸→]\s+/, ''));
      html += '<li>' + content + '</li>';
      continue;
    }

    // ordered list: 1. or 1) (with any leading spaces)
    if (/^\s*\d+[.)]\s+/.test(raw_line)) {
      closeUl();
      if (!inOl) { html += '<ol>'; inOl = true; }
      const content = inlineFormat(raw_line.replace(/^\s*\d+[.)]\s+/, ''));
      html += '<li>' + content + '</li>';
      continue;
    }

    // normal paragraph
    closeLists();
    html += '<p>' + trimmedFormatted + '</p>';
  }

  closeLists();
  return html;
}

function getWelcomeHTML() {
  return `<div class="welcome" id="welcome">
    <div class="welcome-glyph">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
        <path d="M12 2L2 7l10 5 10-5-10-5z"/>
        <path d="M2 17l10 5 10-5"/>
        <path d="M2 12l10 5 10-5"/>
      </svg>
    </div>
    <h2>Your AI Career Advisor</h2>
    <p>Fill in your profile on the left, then ask me anything about careers, colleges, salaries, skills, or entrance exams.</p>
    <div class="welcome-chips">
      <div class="chip" onclick="quickAsk('What careers suit a Science PCM student who loves coding?')">PCM + coding</div>
      <div class="chip" onclick="quickAsk('Best careers after Commerce for a creative person?')">Creative commerce</div>
      <div class="chip" onclick="quickAsk('I have 3 years of IT experience, what is my next move?')">IT next step</div>
      <div class="chip" onclick="quickAsk('What is the scope of Data Science in India in 2025?')">Data Science scope</div>
      <div class="chip" onclick="quickAsk('How do I prepare for JEE, NEET or CAT exam?')">Entrance exam prep</div>
    </div>
  </div>`;
}

function addUserMessage(text) {
  const welcome = document.getElementById('welcome');
  if (welcome) welcome.remove();
  const container = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = 'msg user';
  div.innerHTML = '<div class="msg-avatar">You</div><div class="msg-bubble">' + renderMarkdown(text) + '</div>';
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

function addThinkingMessage() {
  const welcome = document.getElementById('welcome');
  if (welcome) welcome.remove();
  const container = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = 'msg ai';
  div.innerHTML = '<div class="msg-avatar">AI</div><div class="msg-bubble"><div class="thinking"><div class="thinking-dots"><span></span><span></span><span></span></div><span>Thinking...</span></div></div>';
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
  return div.querySelector('.msg-bubble');
}

async function sendMessage(overrideText) {
  const input = document.getElementById('user-input');
  const text = (overrideText || input.value).trim();
  if (!text) return;
  input.value = '';
  autoResize(input);
  document.getElementById('send-btn').disabled = true;
  conversationHistory.push({ role: 'user', content: text });
  addUserMessage(text);
  const bubble = addThinkingMessage();
  try {
    const response = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        messages: conversationHistory.map(m => ({ role: m.role, content: m.content })),
        profile: getProfile()
      })
    });
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let fullText = '';
    bubble.innerHTML = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value, { stream: true });
      for (const line of chunk.split('\n')) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') break;
          try {
            const parsed = JSON.parse(data);
            if (parsed.text) {
              fullText += parsed.text;
              bubble.innerHTML = renderMarkdown(fullText);
              document.getElementById('messages').scrollTop = document.getElementById('messages').scrollHeight;
            }
          } catch(e) {}
        }
      }
    }
    conversationHistory.push({ role: 'assistant', content: fullText });
  } catch(err) {
    bubble.innerHTML = '<p>Something went wrong. Please try again.</p>';
  }
  document.getElementById('send-btn').disabled = false;
}

function quickAsk(text) {
  closeSidebar();
  document.getElementById('user-input').value = text;
  sendMessage();
}

function clearChat() {
  conversationHistory.length = 0;
  document.getElementById('messages').innerHTML = getWelcomeHTML();
}

function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
}

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

renderTags();