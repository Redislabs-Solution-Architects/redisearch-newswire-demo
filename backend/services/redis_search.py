<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NewsWire Search</title>
    <script src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
    <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #ffffff;
            color: #111827;
        }

        .stats-banner {
            background: linear-gradient(135deg, #1e40af 0%, #1e3a8a 100%);
            color: white;
            padding: 16px 20px;
            margin-bottom: 20px;
            border-radius: 8px;
            display: flex;
            justify-content: space-around;
            align-items: center;
            flex-wrap: wrap;
            gap: 20px;
        }

        .stats-item {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 15px;
        }

        .stats-item-value {
            font-weight: 600;
            font-size: 18px;
        }

        .container {
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {
            text-align: center;
            padding: 30px 0 20px 0;
        }

        .header h1 {
            font-size: 28px;
            font-weight: 600;
            color: #111827;
        }

        .header p {
            color: #6b7280;
            font-size: 14px;
            margin-top: 8px;
        }

        .search-row {
            display: flex;
            gap: 16px;
            align-items: center;
            margin-bottom: 20px;
            position: relative;
        }

        .search-box {
            flex: 1;
            position: relative;
        }

        .search-box input {
            width: 100%;
            padding: 14px 20px;
            font-size: 15px;
            border: 1px solid #e5e7eb;
            border-radius: 24px;
            outline: none;
            transition: border-color 0.2s;
        }

        .search-box input:focus {
            border-color: #d1d5db;
        }

        .search-box input::placeholder {
            color: #9ca3af;
        }

        .autocomplete-dropdown {
            position: absolute;
            top: 100%;
            left: 0;
            right: 0;
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
            margin-top: 8px;
            z-index: 1000;
            max-height: 400px;
            overflow-y: auto;
        }

        .autocomplete-section {
            padding: 12px 16px;
        }

        .autocomplete-section-title {
            font-size: 12px;
            font-weight: 600;
            color: #6b7280;
            text-transform: uppercase;
            margin-bottom: 8px;
        }

        .autocomplete-item {
            padding: 10px 12px;
            cursor: pointer;
            border-radius: 6px;
            transition: background 0.2s;
            margin-bottom: 4px;
        }

        .autocomplete-item:hover {
            background: #f3f4f6;
        }

        .autocomplete-item-title {
            font-size: 14px;
            color: #111827;
            font-weight: 500;
            margin-bottom: 4px;
        }

        .autocomplete-item-meta {
            font-size: 12px;
            color: #6b7280;
        }

        .autocomplete-spell {
            padding: 10px 12px;
            background: #fef3c7;
            border-radius: 6px;
            font-size: 14px;
            cursor: pointer;
            margin-bottom: 4px;
        }

        .autocomplete-spell:hover {
            background: #fde68a;
        }

        .autocomplete-footer {
            padding: 8px 16px;
            border-top: 1px solid #e5e7eb;
            font-size: 12px;
            color: #9ca3af;
            text-align: center;
        }

        .recent-btn {
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 10px 16px;
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 20px;
            font-size: 14px;
            color: #374151;
            cursor: pointer;
            transition: background 0.2s;
            position: relative;
        }

        .recent-btn:hover {
            background: #f3f4f6;
        }

        .recent-dropdown {
            position: absolute;
            top: 100%;
            right: 0;
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
            margin-top: 8px;
            z-index: 1000;
            min-width: 320px;
            max-height: 400px;
            overflow-y: auto;
        }

        .recent-dropdown-header {
            padding: 12px 16px;
            border-bottom: 1px solid #e5e7eb;
            font-weight: 600;
            font-size: 14px;
            color: #111827;
        }

        .recent-item {
            padding: 12px 16px;
            border-bottom: 1px solid #f3f4f6;
            cursor: pointer;
            transition: background 0.2s;
        }

        .recent-item:hover {
            background: #f9fafb;
        }

        .recent-item:last-child {
            border-bottom: none;
        }

        .recent-item-query {
            font-size: 14px;
            font-weight: 500;
            color: #111827;
            margin-bottom: 4px;
        }

        .recent-item-filters {
            font-size: 12px;
            color: #6b7280;
            margin-bottom: 2px;
        }

        .recent-item-time {
            font-size: 11px;
            color: #9ca3af;
        }

        .recent-clear {
            padding: 10px 16px;
            text-align: center;
            border-top: 1px solid #e5e7eb;
            cursor: pointer;
            font-size: 13px;
            color: #dc2626;
            transition: background 0.2s;
        }

        .recent-clear:hover {
            background: #fef2f2;
        }

        .recent-empty {
            padding: 24px 16px;
            text-align: center;
            color: #9ca3af;
            font-size: 14px;
        }

        .fuzzy-toggle {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 14px;
            color: #6b7280;
            cursor: pointer;
            user-select: none;
        }

        .fuzzy-toggle input {
            width: 18px;
            height: 18px;
            cursor: pointer;
        }

        .category-pills {
            display: flex;
            gap: 12px;
            padding: 12px 0;
            border-bottom: 1px solid #e5e7eb;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }

        .category-pill {
            position: relative;
            padding: 10px 20px;
            font-size: 14px;
            font-weight: 500;
            color: #6b7280;
            cursor: pointer;
            transition: all 0.2s;
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 20px;
        }

        .category-pill:hover {
            background: #f3f4f6;
            color: #111827;
        }

        .category-pill.active {
            background: #111827;
            color: #ffffff;
            border-color: #111827;
        }

        .category-pill .count {
            display: none;
            position: absolute;
            top: -10px;
            right: -10px;
            background: #ef4444;
            color: white;
            font-size: 10px;
            padding: 3px 8px;
            border-radius: 10px;
            white-space: nowrap;
        }

        .category-pill:hover .count {
            display: block;
        }

        .main-layout {
            display: flex;
            gap: 30px;
        }

        .sidebar {
            width: 200px;
            flex-shrink: 0;
        }

        .filter-section {
            margin-bottom: 24px;
        }

        .filter-section h3 {
            font-size: 14px;
            font-weight: 600;
            color: #111827;
            margin-bottom: 12px;
        }

        .filter-option {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 6px 0;
            font-size: 14px;
            color: #4b5563;
            cursor: pointer;
        }

        .filter-option:hover {
            color: #111827;
        }

        .filter-option input[type="checkbox"],
        .filter-option input[type="radio"] {
            width: 16px;
            height: 16px;
            cursor: pointer;
        }

        .results-container {
            flex: 1;
            min-width: 0;
        }

        .results-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
            font-size: 13px;
            color: #6b7280;
        }

        .performance-metrics {
            background: #f0f9ff;
            border: 1px solid #bfdbfe;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 16px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
        }

        .metric-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .metric-label {
            font-size: 13px;
            color: #1e40af;
            font-weight: 500;
        }

        .metric-value {
            font-size: 16px;
            font-weight: 600;
            color: #1e40af;
        }

        .spell-check {
            background: #fef3c7;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 14px;
            margin-bottom: 16px;
        }

        .spell-check span {
            color: #1d4ed8;
            cursor: pointer;
            font-weight: 500;
        }

        .command-accordion {
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            margin-bottom: 16px;
        }

        .command-header {
            padding: 12px 16px;
            cursor: pointer;
            font-size: 14px;
            color: #4b5563;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .command-header:hover {
            background: #f3f4f6;
        }

        .command-content {
            padding: 12px 16px;
            border-top: 1px solid #e5e7eb;
            font-family: monospace;
            font-size: 13px;
            color: #374151;
            background: #f9fafb;
            word-break: break-all;
        }

        .homepage-section {
            margin-bottom: 40px;
        }

        .section-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }

        .section-title {
            font-size: 20px;
            font-weight: 600;
            color: #111827;
        }

        .view-all-btn {
            color: #3b82f6;
            font-size: 14px;
            cursor: pointer;
            text-decoration: none;
        }

        .view-all-btn:hover {
            text-decoration: underline;
        }

        .cards-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 24px;
        }

        .article-card {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 16px;
            cursor: pointer;
            transition: all 0.2s;
        }

        .article-card:hover {
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
            transform: translateY(-2px);
        }

        .article-card h3 {
            font-size: 16px;
            font-weight: 600;
            color: #111827;
            line-height: 1.4;
            margin-bottom: 12px;
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }

        .article-card h3 mark {
            background: #fef08a;
            padding: 0 2px;
        }

        .article-meta {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            margin-bottom: 12px;
            font-size: 12px;
            color: #6b7280;
        }

        .article-summary {
            color: #4b5563;
            line-height: 1.6;
            font-size: 13px;
            margin-bottom: 12px;
            display: -webkit-box;
            -webkit-line-clamp: 3;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }

        .article-summary mark {
            background: #fef08a;
            padding: 0 2px;
        }

        .results-list .article-card {
            padding: 20px 0;
            border: none;
            border-bottom: 1px solid #f3f4f6;
            border-radius: 0;
        }

        .results-list .article-card:hover {
            transform: none;
            box-shadow: none;
            background: #f9fafb;
        }

        .results-list .article-card h3 {
            font-size: 18px;
        }

        .results-list .article-summary {
            font-size: 14px;
            -webkit-line-clamp: 4;
        }

        .view-article-btn {
            background: #f3f4f6;
            border: 1px solid #e5e7eb;
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 13px;
            color: #374151;
            cursor: pointer;
            transition: background 0.2s;
            display: inline-block;
        }

        .view-article-btn:hover {
            background: #e5e7eb;
        }

        .load-more-btn {
            width: 100%;
            padding: 12px;
            margin-top: 20px;
            background: #f3f4f6;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            font-size: 14px;
            color: #374151;
            cursor: pointer;
            transition: background 0.2s;
        }

        .load-more-btn:hover {
            background: #e5e7eb;
        }

        .article-panel {
            width: 350px;
            flex-shrink: 0;
            border-left: 1px solid #e5e7eb;
            padding-left: 20px;
            max-height: 80vh;
            overflow-y: auto;
        }

        .article-panel-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }

        .close-btn {
            background: #fee2e2;
            color: #dc2626;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 14px;
            cursor: pointer;
            transition: background 0.2s;
        }

        .close-btn:hover {
            background: #fecaca;
        }

        .full-article h2 {
            font-size: 22px;
            font-weight: 600;
            color: #111827;
            line-height: 1.3;
            margin-bottom: 16px;
        }

        .full-article-meta {
            display: flex;
            gap: 16px;
            flex-wrap: wrap;
            margin-bottom: 20px;
            font-size: 13px;
            color: #6b7280;
        }

        .full-article-content {
            color: #374151;
            line-height: 1.8;
            font-size: 15px;
            white-space: pre-wrap;
        }

        .trending-sidebar {
            width: 220px;
            flex-shrink: 0;
            border-left: 1px solid #e5e7eb;
            padding-left: 20px;
        }

        .trending-header {
            font-size: 14px;
            font-weight: 600;
            color: #111827;
            margin-bottom: 12px;
        }

        .trending-item {
            padding: 10px 12px;
            margin-bottom: 8px;
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 13px;
            color: #374151;
        }

        .trending-item:hover {
            background: #f3f4f6;
            border-color: #d1d5db;
        }

        .trending-count {
            display: inline-block;
            background: #ef4444;
            color: white;
            font-size: 11px;
            font-weight: 600;
            padding: 2px 6px;
            border-radius: 4px;
            margin-left: 6px;
        }

        .loading {
            text-align: center;
            padding: 40px;
            color: #6b7280;
        }

        .no-results {
            text-align: center;
            padding: 40px;
            color: #6b7280;
        }

        mark {
            background: #fef08a;
            padding: 0 2px;
            border-radius: 2px;
        }

        .results-list .article-card h3:hover {
            color: #3b82f6;
            text-decoration: underline;
        }

        @media (max-width: 1400px) {
            .cards-grid {
                grid-template-columns: repeat(2, 1fr);
            }

            .trending-sidebar {
                width: 180px;
            }
        }

        @media (max-width: 1200px) {
            .main-layout {
                flex-direction: column;
            }

            .sidebar {
                width: 100%;
            }

            .trending-sidebar {
                width: 100%;
                border-left: none;
                border-top: 1px solid #e5e7eb;
                padding-left: 0;
                padding-top: 20px;
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 12px;
            }

            .article-panel {
                width: 100%;
                border-left: none;
                border-top: 1px solid #e5e7eb;
                padding-left: 0;
                padding-top: 20px;
            }
        }

        @media (max-width: 900px) {
            .cards-grid {
                grid-template-columns: 1fr;
            }
        }

        @media (max-width: 768px) {
            .stats-banner {
                flex-direction: column;
                gap: 12px;
            }

            .cards-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>

<body>
    <div id="root"></div>

    <script type="text/babel">
        const { useState, useEffect, useCallback, useRef } = React;

        const API_URL = 'http://localhost:8000/api';

        function timeAgo(timestamp) {
            const seconds = Math.floor((Date.now() - timestamp) / 1000);
            if (seconds < 60) return 'Just now';
            if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
            if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
            if (seconds < 604800) return `${Math.floor(seconds / 86400)}d ago`;
            return new Date(timestamp).toLocaleDateString();
        }

        class RecentArticles {
            constructor() {
                this.maxItems = 5;
            }

            get() {
                try {
                    const stored = localStorage.getItem('newsWireRecentArticles');
                    return stored ? JSON.parse(stored) : [];
                } catch {
                    return [];
                }
            }

            add(article) {
                const articles = this.get();
                const newArticle = {
                    id: article.id,
                    title: article.title,
                    category: article.category,
                    source: article.source,
                    timestamp: Date.now()
                };

                const filtered = articles.filter(a => a.id !== newArticle.id);
                filtered.unshift(newArticle);
                const limited = filtered.slice(0, this.maxItems);

                try {
                    localStorage.setItem('newsWireRecentArticles', JSON.stringify(limited));
                } catch (e) {
                    console.error('Failed to save recent article:', e);
                }
            }

            clear() {
                try {
                    localStorage.removeItem('newsWireRecentArticles');
                } catch (e) {
                    console.error('Failed to clear recent articles:', e);
                }
            }
        }

        const recentArticlesManager = new RecentArticles();

        function App() {
            const [query, setQuery] = useState('');
            const [categories, setCategories] = useState(['All']);
            const [sources, setSources] = useState([]);
            const [selectedSources, setSelectedSources] = useState([]);
            const [sort, setSort] = useState('relevance');
            const [fuzzy, setFuzzy] = useState(false);
            const [offset, setOffset] = useState(0);

            const [allCategories, setAllCategories] = useState([]);
            const [categoryCounts, setCategoryCounts] = useState({});
            const [results, setResults] = useState([]);
            const [totalResults, setTotalResults] = useState(0);
            const [latency, setLatency] = useState(0);
            const [command, setCommand] = useState('');
            const [spellSuggestion, setSpellSuggestion] = useState(null);

            const [selectedArticle, setSelectedArticle] = useState(null);
            const [showCommand, setShowCommand] = useState(false);
            const [loading, setLoading] = useState(false);

            const [homepageData, setHomepageData] = useState(null);
            const [isSearchMode, setIsSearchMode] = useState(false);

            const [autocompleteResults, setAutocompleteResults] = useState(null);
            const [showAutocomplete, setShowAutocomplete] = useState(false);

            const [recentArticles, setRecentArticles] = useState([]);
            const [showRecentDropdown, setShowRecentDropdown] = useState(false);

            
            
            const [performance, setPerformance] = useState(null);
            const [isSearching, setIsSearching] = useState(false);

            const searchBoxRef = useRef(null);
            const recentBtnRef = useRef(null);

            useEffect(() => {
                setRecentArticles(recentArticlesManager.get());
            }, []);

            useEffect(() => {
                fetch(`${API_URL}/categories`)
                    .then(res => res.json())
                    .then(data => setAllCategories(data.categories || []));

                fetch(`${API_URL}/sources`)
                    .then(res => res.json())
                    .then(data => setSources(data.sources || []));
            }, []);

            useEffect(() => {
                if (!isSearchMode) {
                    fetch(`${API_URL}/homepage`)
                        .then(res => res.json())
                        .then(data => setHomepageData(data))
                        .catch(err => console.error('Homepage error:', err));
                }
            }, [isSearchMode]);

            useEffect(() => {
                if (!query || query.length < 2) {  // ⬅️ Added isSearching check
                    setShowAutocomplete(false);
                    return;
                }

                const timer = setTimeout(() => {
                    fetch(`${API_URL}/autocomplete?q=${encodeURIComponent(query)}&limit=5`)
                        .then(res => res.json())
                        .then(data => {
                            setAutocompleteResults(data);
                            setShowAutocomplete(true);
                        })
                        .catch(err => console.error('Autocomplete error:', err));
                }, 150);

                return () => clearTimeout(timer);
            }, [query]); 

            useEffect(() => {
                function handleClickOutside(event) {
                    if (searchBoxRef.current && !searchBoxRef.current.contains(event.target)) {
                        setShowAutocomplete(false);
                    }
                    if (recentBtnRef.current && !recentBtnRef.current.contains(event.target)) {
                        setShowRecentDropdown(false);
                    }
                }

                document.addEventListener('mousedown', handleClickOutside);
                return () => document.removeEventListener('mousedown', handleClickOutside);
            }, []);

            const doSearch = useCallback(async (newOffset = 0, append = false) => {
                setIsSearchMode(true);
                setLoading(true);
                // setShowAutocomplete(false);

                try {
                    const sourceParam = selectedSources.length > 0 ? selectedSources.join(',') : 'All';
                    const categoryParam = categories.includes('All') ? 'All' : categories.join(',');

                    // Only search - spellcheck is already in autocomplete
                    const searchData = await fetch(`${API_URL}/search?q=${encodeURIComponent(query)}&category=${encodeURIComponent(categoryParam)}&source=${encodeURIComponent(sourceParam)}&sort=${sort}&fuzzy=${fuzzy}&offset=${newOffset}&limit=5`)
                        .then(res => res.json());

                    const spellData = { suggestions: null };

                    // Fire category counts in background (non-blocking)
                    fetch(`${API_URL}/category-counts?q=${encodeURIComponent(query)}&source=${encodeURIComponent(sourceParam)}`)
                        .then(res => res.json())
                        .then(countsData => setCategoryCounts(countsData.counts || {}))
                        .catch(err => console.error('Category counts error:', err));

                    console.log('API Response:', searchData);
                    console.log('Total from API:', searchData.total);
                    console.log('Results count:', searchData.results?.length);

                    if (append) {
                        setResults(prev => [...prev, ...(searchData.results || [])]);
                    } else {
                        setResults(searchData.results || []);
                    }

                    setTotalResults(searchData.total || 0);
                    setLatency(searchData.latency_ms || 0);
                    setCommand(searchData.command || '');
                    setOffset(newOffset);
                    setPerformance(searchData.performance || null);

                    // Handle spell check results
                    if (spellData.suggestions && Object.keys(spellData.suggestions).length > 0) {
                        let corrected = query;
                        for (const [orig, sugg] of Object.entries(spellData.suggestions)) {
                            corrected = corrected.replace(orig, sugg);
                        }
                        if (corrected !== query) {
                            setSpellSuggestion(corrected);
                        } else {
                            setSpellSuggestion(null);
                        }
                    } else {
                        setSpellSuggestion(null);
                    }

                } catch (error) {
                    console.error('Search error:', error);
                }

                setLoading(false);
            }, [query, categories, selectedSources, sort, fuzzy]);

            useEffect(() => {
                if (categories.length > 0 && !categories.includes('All')) {
                    setIsSearchMode(true);
                    doSearch(0, false);
                } else if (selectedSources.length > 0) {
                    setIsSearchMode(true);
                    doSearch(0, false);
                } else if (isSearchMode) {
                    doSearch(0, false);
                }
            }, [categories, selectedSources, sort, fuzzy]);

            

            

            const viewArticle = async (docId) => {
                try {
                    const res = await fetch(`${API_URL}/article/${encodeURIComponent(docId)}`);
                    const data = await res.json();
                    if (data.article) {
                        setSelectedArticle(data.article);

                        // Add to recent articles
                        recentArticlesManager.add({
                            id: docId,
                            title: data.article.title,
                            category: data.article.category,
                            source: data.article.source
                        });
                        setRecentArticles(recentArticlesManager.get());
                    }
                } catch (error) {
                    console.error('Error loading article:', error);
                }
            };

            const loadMore = () => {
                doSearch(offset + 10, true);
            };

            const applySuggestion = () => {
                if (spellSuggestion) {
                    setQuery(spellSuggestion);
                }
            };

            const toggleCategory = (cat) => {
                if (cat === 'All') {
                    setCategories(['All']);
                } else {
                    const newCategories = categories.filter(c => c !== 'All');
                    if (newCategories.includes(cat)) {
                        const filtered = newCategories.filter(c => c !== cat);
                        setCategories(filtered.length > 0 ? filtered : ['All']);
                    } else {
                        setCategories([...newCategories, cat]);
                    }
                }
            };

            const toggleSource = (src) => {
                if (selectedSources.includes(src)) {
                    const filtered = selectedSources.filter(s => s !== src);
                    setSelectedSources(filtered);
                } else {
                    setSelectedSources([...selectedSources, src]);
                }
            };

            const viewAllInCategory = (cat) => {
                setCategories([cat]);
                setIsSearchMode(true);
                setQuery('');
                setTimeout(() => {
                    doSearch(0, false);
                }, 100);
            };

            const searchTrendingTag = (tag) => {
                setQuery(tag);
                setIsSearchMode(true);
            };

            const allCount = Object.values(categoryCounts).reduce((a, b) => a + b, 0);

            const ArticleCard = ({ article, onClick }) => (
                <div className="article-card" onClick={() => onClick(article.id)}>
                    <h3 dangerouslySetInnerHTML={{
                        __html: (article.title || 'No title')
                    }} />
                    <div className="article-meta">
                        <span>📅 {article.published_at || 'N/A'}</span>
                    </div>
                    <p
                        className="article-summary"
                        dangerouslySetInnerHTML={{
                            __html: article.short_summary || 'No summary available.'
                        }}
                    />
                </div>
            );

            return (
                <div className="container">
                    <div className="header">
                        <h1>📰 NewsWire Search</h1>
                        <p>Professional news search powered by Azure Managed Redis (RediSearch)</p>
                    </div>

                    

                    <div className="search-row">
                        <div className="search-box" ref={searchBoxRef}>
                            
                            <input
                                type="text"
                                placeholder="Search for news, topics or sources..."
                                value={query}
                                onChange={(e) => setQuery(e.target.value)}
                                onFocus={() => query.length >= 2 && setShowAutocomplete(true)}
                                onKeyDown={(e) => {
                                    if (e.key === 'Enter') {
                                        setShowAutocomplete(false);
                                        if (query.trim()) {
                                            doSearch(0, false);
                                        }
                                    }
                                }}
                            />

                            {showAutocomplete && autocompleteResults && (
                                <div className="autocomplete-dropdown">
                                    {autocompleteResults.articles && autocompleteResults.articles.length > 0 && (
                                        <div className="autocomplete-section">
                                            <div className="autocomplete-section-title">🔍 Matching Articles</div>
                                            {autocompleteResults.articles.map((article, idx) => (
                                                <div
                                                    key={idx}
                                                    className="autocomplete-item"
                                                    onClick={() => {
                                                        viewArticle(article.id);
                                                        setShowAutocomplete(false);
                                                    }}
                                                >
                                                    <div className="autocomplete-item-title">{article.title}</div>
                                                    <div className="autocomplete-item-meta">
                                                        {article.category} • {article.source} • {article.date}
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    )}

                                    {autocompleteResults.spell_suggestion && (
                                        <div className="autocomplete-section">
                                            <div
                                                className="autocomplete-spell"
                                                onClick={() => {
                                                    setQuery(autocompleteResults.spell_suggestion);
                                                    setShowAutocomplete(false);
                                                }}
                                            >
                                                💡 Did you mean: <strong>{autocompleteResults.spell_suggestion}</strong>?
                                            </div>
                                        </div>
                                    )}

                                    <div className="autocomplete-footer">
                                        Press ↵ to search • ↑↓ to navigate • Esc to close
                                    </div>
                                </div>
                            )}
                        </div>

                        <label className="fuzzy-toggle">
                            <input
                                type="checkbox"
                                checked={fuzzy}
                                onChange={(e) => setFuzzy(e.target.checked)}
                            />
                            🔮 Fuzzy search
                        </label>

                        <div style={{ position: 'relative' }} ref={recentBtnRef}>
                            <button
                                className="recent-btn"
                                onClick={() => setShowRecentDropdown(!showRecentDropdown)}
                            >
                                🕐 Recent {recentArticles.length > 0 && `(${recentArticles.length})`}
                            </button>

                            {showRecentDropdown && (
                                <div className="recent-dropdown">
                                    <div className="recent-dropdown-header">Recently Viewed Articles</div>
                                    {recentArticles.length === 0 ? (
                                        <div className="recent-empty">No recently viewed articles</div>
                                    ) : (
                                        <>
                                            {recentArticles.map((article, idx) => (
                                                <div
                                                    key={idx}
                                                    className="recent-item"
                                                    onClick={() => {
                                                        viewArticle(article.id);
                                                        setShowRecentDropdown(false);
                                                    }}
                                                >
                                                    <div className="recent-item-query">{article.title}</div>
                                                    <div className="recent-item-filters">
                                                        {article.category} • {article.source}
                                                    </div>
                                                    <div className="recent-item-time">{timeAgo(article.timestamp)}</div>
                                                </div>
                                            ))}
                                            <div className="recent-clear" onClick={() => {
                                                recentArticlesManager.clear();
                                                setRecentArticles([]);
                                                setShowRecentDropdown(false);
                                            }}>
                                                Clear History
                                            </div>
                                        </>
                                    )}
                                </div>
                            )}
                        </div>
                    </div>

                    <div className="category-pills">
                        <button
                            className={`category-pill ${categories.includes('All') ? 'active' : ''}`}
                            onClick={() => toggleCategory('All')}
                        >
                            All
                            <span className="count">{allCount.toLocaleString()}</span>
                        </button>
                        {allCategories.slice(0, 12).map(cat => (
                            <button
                                key={cat.name}
                                className={`category-pill ${categories.includes(cat.name) ? 'active' : ''}`}
                                onClick={() => toggleCategory(cat.name)}
                            >
                                {cat.name}
                                <span className="count">
                                    {isSearchMode ? (categoryCounts[cat.name] || 0).toLocaleString() : cat.count.toLocaleString()}
                                </span>
                            </button>
                        ))}
                    </div>

                    <div className="main-layout">
                        <div className="sidebar">
                            <div className="filter-section">
                                <h3>📰 Source</h3>
                                <label className="filter-option">
                                    <input
                                        type="checkbox"
                                        checked={selectedSources.length === 0}
                                        onChange={() => setSelectedSources([])}
                                    />
                                    All
                                </label>
                                {sources.slice(0, 10).map(s => (
                                    <label key={s.name} className="filter-option">
                                        <input
                                            type="checkbox"
                                            checked={selectedSources.includes(s.name)}
                                            onChange={() => toggleSource(s.name)}
                                        />
                                        {s.name}
                                    </label>
                                ))}
                            </div>

                            <div className="filter-section">
                                <h3>📊 Sort By</h3>
                                <label className="filter-option">
                                    <input
                                        type="radio"
                                        name="sort"
                                        checked={sort === 'relevance'}
                                        onChange={() => setSort('relevance')}
                                    />
                                    Relevance
                                </label>
                                <label className="filter-option">
                                    <input
                                        type="radio"
                                        name="sort"
                                        checked={sort === 'date_desc'}
                                        onChange={() => setSort('date_desc')}
                                    />
                                    Newest First
                                </label>
                                <label className="filter-option">
                                    <input
                                        type="radio"
                                        name="sort"
                                        checked={sort === 'date_asc'}
                                        onChange={() => setSort('date_asc')}
                                    />
                                    Oldest First
                                </label>
                            </div>
                        </div>

                        {!isSearchMode && homepageData ? (
                            <div className="results-container">
                                <div className="homepage-section">
                                    <div className="section-header">
                                        <div className="section-title">🔥 Top Stories</div>
                                    </div>
                                    <div className="cards-grid">
                                        {homepageData.top_stories && homepageData.top_stories.map((article, idx) => (
                                            <ArticleCard key={idx} article={article} onClick={viewArticle} />
                                        ))}
                                    </div>
                                </div>

                                {homepageData.category_sections && homepageData.category_sections.map((section, idx) => (
                                    <div key={idx} className="homepage-section">
                                        <div className="section-header">
                                            <div className="section-title">
                                                {section.category === 'Politics' && '🏛️'}
                                                {section.category === 'Business' && '💼'}
                                                {section.category === 'Technology' && '💻'}
                                                {section.category === 'Sports' && '⚽'}
                                                {section.category === 'Science' && '🔬'}
                                                {section.category}
                                            </div>
                                            <a
                                                className="view-all-btn"
                                                onClick={() => viewAllInCategory(section.category)}
                                            >
                                                View All →
                                            </a>
                                        </div>
                                        <div className="cards-grid">
                                            {section.articles && section.articles.map((article, aIdx) => (
                                                <ArticleCard key={aIdx} article={article} onClick={viewArticle} />
                                            ))}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        ) : (
                            <div className="results-container">
                                {isSearchMode && (
                                    <div className="performance-metrics">
                                        {performance ? (
                                            <>
                                                <div className="metric-item">
                                                    <span className="metric-label">⚡ Redis Query</span>
                                                    <span className="metric-value">{performance.redis_execution_ms?.toFixed(3) || '0'}ms</span>
                                                </div>
                                                <div className="metric-item">
                                                    <span className="metric-label">🔨 Query Build</span>
                                                    <span className="metric-value">{performance.query_build_ms?.toFixed(3) || '0'}ms</span>
                                                </div>
                                                <div className="metric-item">
                                                    <span className="metric-label">📦 Result Parsing</span>
                                                    <span className="metric-value">{performance.result_parsing_ms?.toFixed(3) || '0'}ms</span>
                                                </div>
                                                <div className="metric-item">
                                                    <span className="metric-label">📊 Total Endpoint</span>
                                                    <span className="metric-value">{performance.search_service_total_ms?.toFixed(3) || '0'}ms</span>
                                                </div>
                                            </>
                                        ) : (
                                            <div className="metric-item">
                                                <span className="metric-label">⚡ Query Speed</span>
                                                <span className="metric-value">{latency.toFixed(2)}ms</span>
                                            </div>
                                        )}
                                        <div className="metric-item">
                                            <span className="metric-label">📄 Results</span>
                                            <span className="metric-value">{results.length} of {totalResults.toLocaleString()}</span>
                                        </div>
                                    </div>
                                )}

                                {spellSuggestion && (
                                    <div className="spell-check">
                                        💡 Did you mean: <span onClick={applySuggestion}>{spellSuggestion}</span>?
                                    </div>
                                )}

                                <div className="command-accordion">
                                    <div
                                        className="command-header"
                                        onClick={() => setShowCommand(!showCommand)}
                                    >
                                        <span>📝 Redis Command</span>
                                        <span>{showCommand ? '▼' : '▶'}</span>
                                    </div>
                                    {showCommand && (
                                        <div className="command-content">{command}</div>
                                    )}
                                </div>

                                {loading && <div className="loading">Loading...</div>}

                                {!loading && results.length === 0 && (
                                    <div className="no-results">No results found.</div>
                                )}

                                <div className="results-list">
                                    {results.map((article, index) => (
                                        <div key={article.id || index} className="article-card">
                                            <h3
                                                onClick={() => viewArticle(article.id)}
                                                style={{ cursor: 'pointer' }}
                                                dangerouslySetInnerHTML={{
                                                    __html: (article.title || 'No title')
                                                }}
                                            />
                                            <div className="article-meta">
                                                <span>📅 {article.published_at || 'N/A'}</span>
                                            </div>
                                            <p
                                                className="article-summary"
                                                dangerouslySetInnerHTML={{
                                                    __html: article.short_summary || 'No summary available.'
                                                }}
                                            />
                                        </div>
                                    ))}
                                </div>

                                {results.length < totalResults && (
                                    <button className="load-more-btn" onClick={loadMore}>
                                        Load More Results
                                    </button>
                                )}
                            </div>
                        )}

                        

                        {selectedArticle && (
                            <div className="article-panel">
                                <div className="article-panel-header">
                                    <h3>📄 Full Article</h3>
                                    <button
                                        className="close-btn"
                                        onClick={() => setSelectedArticle(null)}
                                    >
                                        ✕ Close
                                    </button>
                                </div>
                                <div className="full-article">
                                    <h2>{selectedArticle.title}</h2>
                                    <div className="full-article-meta">
                                        <span><strong>Category:</strong> {selectedArticle.category}</span>
                                        <span><strong>Source:</strong> {selectedArticle.source}</span>
                                        <span><strong>Author:</strong> {selectedArticle.author}</span>
                                        <span><strong>Published:</strong> {selectedArticle.published_at}</span>
                                        <span><strong>Words:</strong> {selectedArticle.word_count}</span>
                                    </div>
                                    <hr style={{ border: 'none', borderTop: '1px solid #e5e7eb', margin: '20px 0' }} />
                                    <div className="full-article-content">
                                        {selectedArticle.full_content}
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            );
        }

        ReactDOM.createRoot(document.getElementById('root')).render(<App />);
    </script>
</body>

</html>
