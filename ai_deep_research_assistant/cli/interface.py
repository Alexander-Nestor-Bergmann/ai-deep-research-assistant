"""
Rich CLI Interface for AI Deep Research Assistant.

This module provides a sophisticated terminal interface with real-time progress tracking,
interactive query input, beautiful formatting, and comprehensive citation display
for an optimal user experience.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.rule import Rule

# Import our workflow components
try:
    from ..graph.workflow import run_research, create_research_graph
except ImportError:
    # For testing when run directly
    import sys
    import os
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    
    from graph.workflow import run_research, create_research_graph

logger = logging.getLogger(__name__)

# =============================================================================
# CLI CONFIGURATION AND CONSTANTS
# =============================================================================

class CLIConfig:
    """Configuration for CLI interface appearance and behavior."""
    
    # Console settings
    CONSOLE_WIDTH = 120
    PANEL_PADDING = (1, 2)
    
    # Colors and styling
    PRIMARY_COLOR = "cyan"
    SECONDARY_COLOR = "blue"
    SUCCESS_COLOR = "green" 
    WARNING_COLOR = "yellow"
    ERROR_COLOR = "red"
    MUTED_COLOR = "dim"
    
    # Progress tracking
    PROGRESS_REFRESH_RATE = 10  # Updates per second
    SPINNER_STYLE = "dots12"
    
    # History settings
    MAX_HISTORY_ITEMS = 20
    HISTORY_FILE = ".query_history"

# =============================================================================
# RICH CLI INTERFACE CLASS
# =============================================================================

class ResearchInterface:
    """
    Rich terminal interface for the AI Deep research assistant.
    
    Provides interactive query input, real-time progress tracking, beautiful
    output formatting, and comprehensive result display with citations.
    """
    
    def __init__(self, enable_debug: bool = False, quick_mode: bool = False):
        """
        Initialize the CLI interface.
        
        Args:
            enable_debug: Whether to enable debug mode with verbose output
            quick_mode: Whether to use quick mode for faster research
        """
        self.console = Console(width=CLIConfig.CONSOLE_WIDTH)
        self.enable_debug = enable_debug
        self.quick_mode = quick_mode
        self.query_history: List[str] = []
        self.compiled_workflow = None
        # Persistent thread_id for session continuity 
        self.thread_id = f"session-{int(time.time())}"
        
        # Load query history
        self._load_history()
        
        # Initialize workflow
        self._initialize_workflow()
    
    def _initialize_workflow(self):
        """Initialize the research workflow."""
        try:
            # Workflow will use default_graph when no graph is provided
            self.compiled_workflow = True
            if self.enable_debug:
                self.console.print(f"[{CLIConfig.MUTED_COLOR}]Research workflow initialized for session: {self.thread_id}[/]")
        except Exception as e:
            self.console.print(f"[{CLIConfig.ERROR_COLOR}]Error initializing workflow: {e}[/]")
    
    def _load_history(self):
        """Load query history from file."""
        try:
            import os
            if os.path.exists(CLIConfig.HISTORY_FILE):
                with open(CLIConfig.HISTORY_FILE, 'r') as f:
                    self.query_history = [line.strip() for line in f.readlines()]
                    self.query_history = self.query_history[-CLIConfig.MAX_HISTORY_ITEMS:]
        except Exception as e:
            if self.enable_debug:
                self.console.print(f"[{CLIConfig.MUTED_COLOR}]Could not load history: {e}[/]")
    
    def _save_history(self):
        """Save query history to file."""
        try:
            with open(CLIConfig.HISTORY_FILE, 'w') as f:
                for query in self.query_history[-CLIConfig.MAX_HISTORY_ITEMS:]:
                    f.write(f"{query}\n")
        except Exception as e:
            if self.enable_debug:
                self.console.print(f"[{CLIConfig.MUTED_COLOR}]Could not save history: {e}[/]")
    
    def display_welcome(self):
        """Display welcome message and instructions."""
        mode_indicator = " ⚡ QUICK MODE" if self.quick_mode else ""
        welcome_text = f"""
        # 🔍 AI Deep Research Assistant{mode_indicator}
        
        Welcome to your AI-powered research companion! I can help you with:
        
        • **Comprehensive Research** - Multi-source analysis with academic rigor
        • **Current Events** - Latest news and developments with recency focus  
        • **Technical Analysis** - In-depth exploration of complex topics
        • **Conversational Help** - Quick questions and friendly assistance
        
        **Commands:**
        • Type your question to start research
        • `history` - View recent queries
        • `help` - Show detailed help
        • `quit` or `exit` - Exit the application
        """
        
        welcome_panel = Panel(
            Markdown(welcome_text),
            title="[bold cyan]Research Assistant[/bold cyan]",
            border_style=CLIConfig.PRIMARY_COLOR,
            padding=CLIConfig.PANEL_PADDING
        )
        
        self.console.print(welcome_panel)
        self.console.print()
    
    def display_help(self):
        """Display detailed help information."""
        help_text = """
        # 📖 Detailed Help
        
        ## Research Capabilities
        - **Academic Research**: Scholarly sources, peer-reviewed papers, technical documentation
        - **News Research**: Recent developments, breaking news, trend analysis  
        - **General Research**: Broad web search across diverse reliable sources
        - **Synthesis**: Multi-source integration with confidence scoring and citations
        
        ## Query Examples
        - "What are the latest developments in quantum computing?"
        - "Compare renewable energy policies in different countries"
        - "Explain the impact of AI on healthcare outcomes"
        - "What happened in the recent climate summit?"
        
        ## Features  
        - **Real-time Progress**: See research agents working in parallel
        - **Source Citations**: All claims backed by verifiable sources
        - **Confidence Scoring**: Transparent reliability assessment
        - **Follow-up Suggestions**: Explore related topics
        
        ## Commands
        - `history` - View your recent queries
        - `clear` - Clear the screen
        - `quit` / `exit` - Exit the application
        """
        
        help_panel = Panel(
            Markdown(help_text),
            title="[bold blue]Help & Documentation[/bold blue]",
            border_style=CLIConfig.SECONDARY_COLOR,
            padding=CLIConfig.PANEL_PADDING
        )
        
        self.console.print(help_panel)
    
    def display_history(self):
        """Display query history."""
        if not self.query_history:
            self.console.print(f"[{CLIConfig.MUTED_COLOR}]No query history available.[/]")
            return
        
        history_table = Table(title="Recent Query History", show_header=True, header_style="bold cyan")
        history_table.add_column("#", style="dim", width=4)
        history_table.add_column("Query", min_width=50)
        history_table.add_column("Length", justify="right", width=8)
        
        for i, query in enumerate(self.query_history[-10:], 1):  # Show last 10
            history_table.add_row(str(i), query, str(len(query)))
        
        self.console.print(history_table)
    
    def get_user_input(self) -> Optional[str]:
        """
        Get user input with rich prompt and history support.
        
        Returns:
            User query string or None if quit command
        """
        try:
            # Show prompt with styling
            query = Prompt.ask(
                f"[{CLIConfig.PRIMARY_COLOR}]🔍 Research Query[/]",
                console=self.console
            ).strip()
            
            # Handle special commands
            if query.lower() in ['quit', 'exit']:
                return None
            elif query.lower() == 'help':
                self.display_help()
                return self.get_user_input()
            elif query.lower() == 'history':
                self.display_history()
                return self.get_user_input()
            elif query.lower() == 'clear':
                self.console.clear()
                self.display_welcome()
                return self.get_user_input()
            elif not query:
                return self.get_user_input()
            
            # Add to history
            if query not in self.query_history:
                self.query_history.append(query)
                self._save_history()
            
            return query
            
        except KeyboardInterrupt:
            self.console.print(f"\n[{CLIConfig.WARNING_COLOR}]Research interrupted by user[/]")
            return None
        except EOFError:
            return None
    
    async def execute_research_with_progress(self, query: str) -> Dict[str, Any]:
        """
        Execute research with real-time progress display.
        
        Args:
            query: Research query to process
            
        Returns:
            Research results dictionary
        """
        if not self.compiled_workflow:
            raise ValueError("Workflow not initialized")
        
        session_id = f"cli-{int(time.time())}"
        request_id = f"{session_id}-{int(time.time())}"
        
        # Initialize progress tracking
        with Progress(
            SpinnerColumn(spinner_name=CLIConfig.SPINNER_STYLE),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=self.console,
            refresh_per_second=CLIConfig.PROGRESS_REFRESH_RATE
        ) as progress:
            
            # Add main progress task
            main_task = progress.add_task("🔍 Starting research...", total=None)
            
            # Track execution time from the start
            start_time = time.time()
            
            # Execute research with our simplified system
            try:
                progress.update(main_task, description="🚦 Analyzing query...")
                
                # Use consistent thread_id for session continuity (graph=None uses default_graph)
                result = await run_research(
                    query=query,
                    session_id=session_id, 
                    request_id=request_id,
                    quick_mode=self.quick_mode,
                    graph=None,  # Uses default_graph from workflow.py
                    session_thread_id=self.thread_id  # Use persistent thread_id for session continuity
                )
                
                # Calculate execution time
                execution_time = time.time() - start_time
                
                progress.update(main_task, description="✅ Research complete!")
                
                # Convert to the format expected by the CLI
                if result.get('final_synthesis'):
                    synthesis = result['final_synthesis']
                    # Handle both dict and Pydantic model formats
                    if hasattr(synthesis, 'final_answer'):
                        # Pydantic model format
                        formatted_result = {
                            'response': synthesis.final_answer,
                            'sources': synthesis.source_urls,
                            'confidence': synthesis.confidence_score,
                            'execution_time': execution_time,
                            'key_findings': synthesis.key_findings,
                            'limitations': synthesis.limitations,
                            'workflow_success': True,  # Successfully got synthesis
                            'workflow_type': 'conversation' if result.get('skip_research') else 'research',
                            'error_details': result.get('error_message')
                        }
                    else:
                        # Dict format
                        formatted_result = {
                            'response': synthesis.get('final_answer', 'No response generated'),
                            'sources': synthesis.get('source_urls', []),
                            'confidence': synthesis.get('confidence_score', 0.0),
                            'execution_time': execution_time,
                            'key_findings': synthesis.get('key_findings', []),
                            'limitations': synthesis.get('limitations', []),
                            'workflow_success': True,  # Successfully got synthesis
                            'workflow_type': 'conversation' if result.get('skip_research') else 'research',
                            'error_details': result.get('error_message')
                        }
                else:
                    # Fallback if something went wrong
                    formatted_result = {
                        'response': result.get('conversation_response', 'Sorry, I could not process your request.'),
                        'sources': [],
                        'confidence': 0.0,
                        'execution_time': execution_time,
                        'key_findings': [],
                        'limitations': ['Research workflow encountered an issue'],
                        'workflow_success': False,
                        'workflow_type': 'error',
                        'error_details': result.get('error_message', 'Unknown workflow issue')
                    }
                
                return formatted_result
                
            except Exception as e:
                # Calculate execution time even for errors
                execution_time = time.time() - start_time
                progress.update(main_task, description=f"❌ Research failed: {str(e)[:50]}")
                # Return error result instead of raising to keep CLI running
                return {
                    'response': f'Sorry, I encountered an error: {str(e)}',
                    'sources': [],
                    'confidence': 0.0,
                    'execution_time': execution_time,
                    'key_findings': [],
                    'limitations': ['Error occurred during research']
                }
    
    def display_research_results(self, result: Dict[str, Any], query: str):
        """
        Display comprehensive research results with beautiful formatting.
        
        Args:
            result: Research results dictionary
            query: Original query for context
        """
        # Main response panel
        response = result.get('response', 'No response generated')
        confidence = result.get('confidence', 0.0)
        
        # Format confidence with color coding
        if confidence >= 0.8:
            confidence_color = CLIConfig.SUCCESS_COLOR
            confidence_icon = "🟢"
        elif confidence >= 0.6:
            confidence_color = CLIConfig.WARNING_COLOR  
            confidence_icon = "🟡"
        else:
            confidence_color = CLIConfig.ERROR_COLOR
            confidence_icon = "🔴"
        
        # Create main response panel
        response_panel = Panel(
            Markdown(response),
            title=f"[bold white]Research Response - {confidence_icon} {confidence:.1%} Confidence[/bold white]",
            border_style=confidence_color,
            padding=CLIConfig.PANEL_PADDING
        )
        
        self.console.print(response_panel)
        self.console.print()
        
        # Research metadata
        self._display_research_metadata(result)
        
        # Citations and sources
        sources = result.get('sources', [])
        if sources:
            self._display_citations(sources)
        
        # Follow-up suggestions
        follow_ups = result.get('follow_up_suggestions', [])
        if follow_ups:
            self._display_follow_ups(follow_ups)
        
        # Research summary
        self._display_research_summary(result)
    
    def _display_research_metadata(self, result: Dict[str, Any]):
        """Display research execution metadata."""
        metadata_table = Table(show_header=False, box=None, padding=(0, 2))
        metadata_table.add_column("Label", style="bold cyan", width=20)
        metadata_table.add_column("Value", style="white")
        
        # Add metadata rows
        execution_time = result.get('execution_time', 0)
        metadata_table.add_row("⏱️  Execution Time", f"{execution_time:.2f} seconds")
        
        # More informative status display
        workflow_type = result.get('workflow_type', 'unknown')
        error_details = result.get('error_details')
        
        if result.get('workflow_success'):
            if workflow_type == 'conversation':
                metadata_table.add_row("✅ Status", "Conversation Response")
            elif workflow_type == 'research':
                metadata_table.add_row("✅ Status", "Research Complete")
            else:
                metadata_table.add_row("✅ Status", "Success")
        else:
            if error_details:
                metadata_table.add_row("❌ Status", f"Failed: {error_details[:50]}...")
            else:
                metadata_table.add_row("❌ Status", "Partial Success")
        
        research_summary = result.get('research_summary', '')
        if research_summary:
            metadata_table.add_row("📊 Research", research_summary)
        
        sources_count = len(result.get('sources', []))
        if sources_count > 0:
            metadata_table.add_row("📚 Sources", f"{sources_count} sources analyzed")
        
        self.console.print(metadata_table)
        self.console.print()
    
    def _extract_domain_from_url(self, url: str) -> str:
        """Extract a clean domain name from a URL for display."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc
            # Remove www. prefix if present
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain or 'Unknown Source'
        except Exception:
            return 'Unknown Source'
    
    def _display_citations(self, sources: List):
        """Display sources and citations in a formatted table.
        
        Handles both URL strings and dict objects with metadata.
        """
        if not sources:
            return
        
        citations_table = Table(title="📚 Sources & Citations", show_header=True, header_style="bold cyan")
        citations_table.add_column("#", width=3, justify="right", style="dim")
        citations_table.add_column("Title", min_width=40, max_width=60)
        citations_table.add_column("URL", min_width=30, max_width=50, style="blue underline")
        citations_table.add_column("Relevance", width=10, justify="center")
        
        for i, source in enumerate(sources[:10], 1):  # Limit to top 10 sources
            # Handle both string URLs and dict objects
            if isinstance(source, str):
                # Simple URL string
                url = source
                title = self._extract_domain_from_url(url)
                relevance = 0.8  # Default relevance for URL-only sources
            else:
                # Dictionary with metadata
                title = source.get('title', 'Unknown Title')
                url = source.get('url', '')
                relevance = source.get('relevance', 0.0)
            
            # Truncate long titles
            if len(title) > 60:
                title = title[:57] + "..."
            
            # Format relevance score
            if relevance >= 0.8:
                relevance_text = f"[green]{relevance:.2f}[/green]"
            elif relevance >= 0.6:
                relevance_text = f"[yellow]{relevance:.2f}[/yellow]" 
            else:
                relevance_text = f"[red]{relevance:.2f}[/red]"
            
            citations_table.add_row(
                str(i),
                title,
                url,
                relevance_text
            )
        
        self.console.print(citations_table)
        self.console.print()
    
    def _display_follow_ups(self, follow_ups: List[str]):
        """Display follow-up suggestions."""
        if not follow_ups:
            return
        
        followup_items = []
        for i, suggestion in enumerate(follow_ups[:5], 1):  # Limit to 5 suggestions
            followup_items.append(f"[bold cyan]{i}.[/bold cyan] {suggestion}")
        
        followup_panel = Panel(
            "\n".join(followup_items),
            title="[bold yellow]💡 Follow-up Suggestions[/bold yellow]",
            border_style=CLIConfig.WARNING_COLOR,
            padding=CLIConfig.PANEL_PADDING
        )
        
        self.console.print(followup_panel)
        self.console.print()
    
    def _display_research_summary(self, result: Dict[str, Any]):
        """Display execution summary and performance metrics."""
        errors = result.get('errors', [])
        metrics = result.get('execution_metrics', [])
        
        summary_info = []
        
        # Execution summary
        # More informative summary based on workflow type
        workflow_type = result.get('workflow_type', 'unknown')
        error_details = result.get('error_details')
        
        if result.get('workflow_success'):
            if workflow_type == 'conversation':
                summary_info.append("✅ Conversational response provided")
            elif workflow_type == 'research':
                summary_info.append("✅ Research completed successfully")
            else:
                summary_info.append("✅ Request completed successfully")
        else:
            if error_details:
                summary_info.append(f"❌ Request failed: {error_details}")
            else:
                summary_info.append("⚠️  Request completed with some issues")
        
        # Performance info
        execution_time = result.get('execution_time', 0)
        if execution_time > 0:
            summary_info.append(f"⏱️  Completed in {execution_time:.1f} seconds")
        
        # Agent info
        if metrics:
            unique_nodes = len(set(m.get('node_name', '') for m in metrics))
            summary_info.append(f"🤖 Used {unique_nodes} research agents")
        
        # Error info
        if errors:
            summary_info.append(f"⚠️  {len(errors)} minor issues encountered")
        
        if summary_info:
            summary_text = "\n".join(summary_info)
            summary_panel = Panel(
                summary_text,
                title="[bold green]📊 Execution Summary[/bold green]",
                border_style=CLIConfig.SUCCESS_COLOR,
                padding=CLIConfig.PANEL_PADDING
            )
            
            self.console.print(summary_panel)
    
    def display_error(self, error: Exception, query: str = ""):
        """Display error message with helpful information."""
        error_text = f"""
        # ❌ Research Error
        
        An error occurred while processing your query:
        
        **Error Details:**
        ```
        {str(error)}
        ```
        
        **Suggestions:**
        - Try rephrasing your query
        - Check your internet connection  
        - Use `help` for query examples
        - Contact support if the issue persists
        """
        
        error_panel = Panel(
            Markdown(error_text),
            title="[bold red]Error[/bold red]",
            border_style=CLIConfig.ERROR_COLOR,
            padding=CLIConfig.PANEL_PADDING
        )
        
        self.console.print(error_panel)
    
    async def run_interactive_session(self):
        """
        Run the main interactive CLI session.
        
        Handles user input, executes research, and displays results in a loop
        until the user chooses to exit.
        """
        self.console.clear()
        self.display_welcome()
        
        while True:
            try:
                # Get user query
                query = self.get_user_input()
                
                if query is None:
                    # User wants to quit
                    self.console.print(f"[{CLIConfig.SUCCESS_COLOR}]Thank you for using the Research Assistant! 👋[/]")
                    break
                
                # Execute research with progress tracking
                self.console.print(f"\n[{CLIConfig.PRIMARY_COLOR}]Starting research on:[/] {query}")
                self.console.print()
                
                start_time = time.time()
                result = await self.execute_research_with_progress(query)
                end_time = time.time()
                
                # Display results
                self.console.print(Rule(style=CLIConfig.PRIMARY_COLOR))
                self.display_research_results(result, query)
                
                # Execution summary
                total_time = end_time - start_time
                self.console.print(Rule(style=CLIConfig.MUTED_COLOR))
                self.console.print(f"[{CLIConfig.MUTED_COLOR}]Total session time: {total_time:.2f} seconds[/]")
                self.console.print()
                
            except KeyboardInterrupt:
                self.console.print(f"\n[{CLIConfig.WARNING_COLOR}]Session interrupted by user[/]")
                continue
            
            except Exception as e:
                logger.error(f"Error in interactive session: {e}")
                self.display_error(e, query if 'query' in locals() else "")
                self.console.print()

# =============================================================================
# CLI TESTING AND UTILITIES
# =============================================================================

async def test_cli_interface():
    """Test the CLI interface functionality."""
    print("🧪 Testing CLI Interface...")
    
    try:
        # Test interface initialization
        interface = ResearchInterface(enable_debug=True)
        
        if interface.compiled_workflow:
            print("✅ Interface initialized successfully")
        else:
            print("❌ Interface initialization failed")
            return False
        
        # Test simple research execution (without interactive session)
        test_query = "What is machine learning?"
        print(f"🔍 Testing research execution: {test_query}")
        
        result = await interface.execute_research_with_progress(test_query)
        
        if result and result.get('response'):
            print("✅ Research execution successful")
            print(f"   Response length: {len(result['response'])} characters")
            print(f"   Execution time: {result.get('execution_time', 0):.2f}s")
            print(f"   Sources: {len(result.get('sources', []))}")
        else:
            print("❌ Research execution failed")
            return False
        
        print("🎉 CLI interface testing completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ CLI interface testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_cli_interface(enable_debug: bool = False, quick_mode: bool = False) -> ResearchInterface:
    """
    Factory function to create a configured CLI interface.
    
    Args:
        enable_debug: Whether to enable debug mode
        quick_mode: Whether to use quick mode for faster research
        
    Returns:
        Configured ResearchInterface instance
    """
    return ResearchInterface(enable_debug=enable_debug, quick_mode=quick_mode)


if __name__ == "__main__":
    """Run CLI interface when executed directly"""
    import sys
    
    async def main():
        if len(sys.argv) > 1 and sys.argv[1] == "test":
            # Run tests
            success = await test_cli_interface()
            sys.exit(0 if success else 1)
        else:
            # Run interactive session
            interface = ResearchInterface(enable_debug=False)
            await interface.run_interactive_session()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ CLI failed to start: {e}")
        sys.exit(1)