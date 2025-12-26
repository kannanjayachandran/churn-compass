import React from 'react';
import { Header } from './Header';
import { Footer } from './Footer';

interface ShellProps {
    children: React.ReactNode;
}

export const Shell: React.FC<ShellProps> = ({ children }) => {
    return (
        <div className="flex min-h-screen flex-col bg-background font-sans antialiased">
            <Header />
            <main className="flex-1 flex flex-col">
                {children}
            </main>
            <Footer />
        </div>
    );
};
