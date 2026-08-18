import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import Navbar from './components/Navbar';
import DocumentPanel from './components/DocumentPanel';
import ChatInterface from './components/ChatInterface';
import api from './services/api';

export default function App() {
  const [conversationId, setConversationId] = useState(null);

  const { data: stats } = useQuery({
    queryKey: ['stats'],
    queryFn: api.stats,
    refetchInterval: 10000,
  });

  return (
    <div className="flex h-full flex-col">
      <Navbar stats={stats} />
      <div className="mx-auto flex w-full max-w-7xl flex-1 gap-4 overflow-hidden p-4">
        <aside className="w-80 shrink-0 overflow-hidden">
          <DocumentPanel />
        </aside>
        <main className="flex-1 overflow-hidden">
          <ChatInterface
            conversationId={conversationId}
            onConversationCreate={setConversationId}
          />
        </main>
      </div>
    </div>
  );
}
