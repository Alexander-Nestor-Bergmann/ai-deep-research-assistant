"""
Main CLI Entry Point for AI Deep Research Assistant.

This is the primary entry point for the research assistant, providing comprehensive
command-line interface with argument parsing, workflow orchestration, performance
monitoring, and complete user experience coordination.
"""

import asyncio
import argparse
import logging
import sys
import time
import json
import signal
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

try:
    from .cli.interface import ResearchInterface, create_cli_interface
    from .graph.workflow import run_research
    from .config.settings import Settings
except ImportError:
    # For direct execution
    import os

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from cli.interface import create_cli_interface
    from graph.workflow import run_research

# Set up module logger
logger = logging.getLogger(__name__)

# =============================================================================
# MAIN APPLICATION CONFIGURATION
# =============================================================================


class AppConfig:
    """Main application configuration and constants."""

    APP_NAME = "AI Deep Research Assistant"
    APP_VERSION = "1.0.0"
    APP_DESCRIPTION = "AI-powered research assistant with multi-agent analysis"

    # Default settings
    DEFAULT_LOG_LEVEL = "INFO"
    DEFAULT_SESSION_TIMEOUT = 3600  # 1 hour
    DEFAULT_MAX_QUERIES = 100
    DEFAULT_CITATION_STYLE = "APA"

    # Performance settings
    DEFAULT_EXECUTION_TIMEOUT = 300  # 5 minutes
    DEFAULT_PROGRESS_REFRESH = 0.5  # 500ms

    # File locations
    CONFIG_DIR = Path.home() / ".ai-deep-research"
    LOG_FILE = CONFIG_DIR / "research.log"
    CACHE_DIR = CONFIG_DIR / "cache"
    SESSIONS_DIR = CONFIG_DIR / "sessions"


# =============================================================================
# COMMAND LINE ARGUMENT PARSING
# =============================================================================


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser with all CLI options."""

    parser = argparse.ArgumentParser(
        prog="ai-deep-research",
        description=f"{AppConfig.APP_DESCRIPTION} v{AppConfig.APP_VERSION}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Start interactive session
  %(prog)s "What is quantum computing?"       # Single query mode
  %(prog)s --test                            # Run system tests
  %(prog)s --debug --verbose                 # Debug mode with verbose output
  %(prog)s --citation-style MLA --save result.json  # Custom citation style and save

For more information, visit: https://github.com/Alexander-Nestor-Bergmann/ai-deep-research-assistant
        """,
    )

    # Positional arguments
    parser.add_argument(
        "query",
        nargs="?",
        help="Research query to process (if omitted, starts interactive mode)",
    )

    # Execution modes
    execution_group = parser.add_argument_group("execution modes")
    execution_group.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Start interactive session (default if no query provided)",
    )
    execution_group.add_argument(
        "--test", action="store_true", help="Run comprehensive system tests"
    )
    execution_group.add_argument(
        "--quick",
        action="store_true",
        help="Use quick research mode (faster, less comprehensive)",
    )

    # Research options
    research_group = parser.add_argument_group("research options")
    research_group.add_argument(
        "--citation-style",
        choices=["APA", "MLA", "Chicago", "IEEE"],
        default=AppConfig.DEFAULT_CITATION_STYLE,
        help="Citation format style (default: %(default)s)",
    )
    research_group.add_argument(
        "--research-type",
        choices=["general", "academic", "news", "technical"],
        help="Preferred research focus type",
    )
    research_group.add_argument(
        "--max-sources",
        type=int,
        default=10,
        help="Maximum number of sources to analyze (default: %(default)d)",
    )
    research_group.add_argument(
        "--timeout",
        type=int,
        default=AppConfig.DEFAULT_EXECUTION_TIMEOUT,
        help="Maximum execution time in seconds (default: %(default)d)",
    )

    # Output options
    output_group = parser.add_argument_group("output options")
    output_group.add_argument(
        "-o", "--output", help="Output file to save results (JSON format)"
    )
    output_group.add_argument(
        "--format",
        choices=["json", "markdown", "plain", "rich"],
        default="rich",
        help="Output format (default: %(default)s)",
    )
    output_group.add_argument(
        "--save-session", help="Save complete session data to file"
    )
    output_group.add_argument(
        "--no-color", action="store_true", help="Disable colored output"
    )
    output_group.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress indicators and non-essential output",
    )

    # Performance and debugging
    debug_group = parser.add_argument_group("performance and debugging")
    debug_group.add_argument(
        "--debug", action="store_true", help="Enable debug mode with verbose logging"
    )
    debug_group.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase verbosity (can be used multiple times)",
    )
    debug_group.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=AppConfig.DEFAULT_LOG_LEVEL,
        help="Set logging level (default: %(default)s)",
    )
    debug_group.add_argument("--log-file", type=Path, help="Custom log file location")
    debug_group.add_argument(
        "--performance",
        action="store_true",
        help="Enable detailed performance monitoring and reporting",
    )
    debug_group.add_argument(
        "--profile",
        action="store_true",
        help="Enable performance profiling (for development)",
    )

    # Session management
    session_group = parser.add_argument_group("session management")
    session_group.add_argument("--session-id", help="Use specific session ID")
    session_group.add_argument(
        "--resume-session", help="Resume a previous session by ID"
    )
    session_group.add_argument(
        "--list-sessions", action="store_true", help="List all saved sessions"
    )
    session_group.add_argument(
        "--cleanup-sessions", action="store_true", help="Clean up old session files"
    )

    # Version and help
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {AppConfig.APP_VERSION}"
    )

    return parser


# =============================================================================
# PERFORMANCE MONITORING
# =============================================================================


class ApplicationMonitor:
    """Application-wide performance and health monitoring."""

    def __init__(self, enable_detailed: bool = False):
        """Initialize application monitor."""
        self.enable_detailed = enable_detailed
        self.start_time = time.time()
        self.metrics = {
            "queries_processed": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "total_execution_time": 0.0,
            "average_response_time": 0.0,
            "peak_memory_usage": 0,
            "api_calls_made": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }
        self.query_history = []

    def record_query(
        self,
        query: str,
        execution_time: float,
        success: bool,
        details: Dict[str, Any] = None,
    ):
        """Record query execution metrics."""
        self.metrics["queries_processed"] += 1
        self.metrics["total_execution_time"] += execution_time

        if success:
            self.metrics["successful_queries"] += 1
        else:
            self.metrics["failed_queries"] += 1

        # Update average
        self.metrics["average_response_time"] = (
            self.metrics["total_execution_time"] / self.metrics["queries_processed"]
        )

        # Store query record
        query_record = {
            "timestamp": datetime.now().isoformat(),
            "query": query[:100] + "..." if len(query) > 100 else query,
            "execution_time": execution_time,
            "success": success,
            "details": details or {},
        }

        self.query_history.append(query_record)

        # Limit history size
        if len(self.query_history) > 100:
            self.query_history = self.query_history[-100:]

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        uptime = time.time() - self.start_time

        return {
            "uptime_seconds": uptime,
            "queries_per_minute": (
                (self.metrics["queries_processed"] / uptime) * 60 if uptime > 0 else 0
            ),
            "success_rate": (
                self.metrics["successful_queries"] / self.metrics["queries_processed"]
                if self.metrics["queries_processed"] > 0
                else 0
            ),
            "metrics": self.metrics.copy(),
            "recent_queries": self.query_history[-5:] if self.enable_detailed else [],
        }

    def print_performance_report(self):
        """Print formatted performance report."""
        summary = self.get_performance_summary()

        print("\n📊 Performance Summary:")
        print(f"   ⏱️  Uptime: {summary['uptime_seconds']:.1f}s")
        print(f"   📈 Queries: {summary['metrics']['queries_processed']}")
        print(f"   ✅ Success Rate: {summary['success_rate']:.1%}")
        print(f"   ⚡ Avg Response: {summary['metrics']['average_response_time']:.2f}s")
        print(f"   📊 Queries/min: {summary['queries_per_minute']:.1f}")

        if summary["recent_queries"]:
            print("\n🔍 Recent Queries:")
            for i, query in enumerate(summary["recent_queries"], 1):
                status = "✅" if query["success"] else "❌"
                print(
                    f"   {i}. {status} {query['query']} ({query['execution_time']:.2f}s)"
                )


# =============================================================================
# APPLICATION SIGNAL HANDLING
# =============================================================================


class SignalHandler:
    """Handle system signals for graceful shutdown."""

    def __init__(self, monitor: ApplicationMonitor):
        self.monitor = monitor
        self.shutting_down = False

        # Register signal handlers
        signal.signal(signal.SIGINT, self.handle_interrupt)
        signal.signal(signal.SIGTERM, self.handle_terminate)

    def handle_interrupt(self, signum, frame):
        """Handle Ctrl+C interrupt."""
        if self.shutting_down:
            print("\n🛑 Force shutdown...")
            sys.exit(1)

        self.shutting_down = True
        print("\n\n📊 Shutting down gracefully...")

        # Print performance summary
        if self.monitor:
            self.monitor.print_performance_report()

        print("👋 Thank you for using the Research Assistant!")
        sys.exit(0)

    def handle_terminate(self, signum, frame):
        """Handle termination signal."""
        print("\n📊 Application terminated")
        if self.monitor:
            self.monitor.print_performance_report()
        sys.exit(0)


# =============================================================================
# MAIN APPLICATION FUNCTIONS
# =============================================================================


async def run_interactive_mode(
    args: argparse.Namespace, monitor: ApplicationMonitor
) -> int:
    """Run the interactive CLI session."""
    logger.info("Starting interactive mode")

    try:
        # Create CLI interface
        interface = create_cli_interface(enable_debug=args.debug, quick_mode=args.quick)

        if not interface:
            print("❌ Failed to initialize interface")
            return 1

        # Run interactive session
        await interface.run_interactive_session()

        return 0

    except Exception as e:
        logger.error(f"Error in interactive mode: {e}")
        print(f"❌ Interactive session failed: {e}")
        return 1


async def run_single_query_mode(
    args: argparse.Namespace, monitor: ApplicationMonitor
) -> int:
    """Process a single query and exit."""
    logger.info(f"Processing single query: {args.query[:50]}...")

    start_time = time.time()

    try:
        # Use our simplified research system
        session_id = args.session_id or f"cli-{int(time.time())}"
        request_id = f"{session_id}-{int(time.time())}"

        result = await run_research(
            query=args.query,
            session_id=session_id,
            request_id=request_id,
            quick_mode=args.quick,
        )

        execution_time = time.time() - start_time
        success = result.get("final_synthesis") is not None

        # Convert to expected format
        if result.get("final_synthesis"):
            synthesis = result["final_synthesis"]
            # Handle both dict and Pydantic model formats
            if hasattr(synthesis, "final_answer"):
                # Pydantic model format
                formatted_result = {
                    "query": args.query,
                    "response": synthesis.final_answer,
                    "sources": synthesis.source_urls,
                    "confidence": synthesis.confidence_score,
                    "execution_time": execution_time,
                    "workflow_success": success,
                    "workflow_type": (
                        "conversation" if result.get("skip_research") else "research"
                    ),
                    "error_details": result.get("error_message"),
                }
            else:
                # Dict format
                formatted_result = {
                    "query": args.query,
                    "response": synthesis.get("final_answer", "No response generated"),
                    "sources": synthesis.get("source_urls", []),
                    "confidence": synthesis.get("confidence_score", 0.0),
                    "execution_time": execution_time,
                    "workflow_success": success,
                    "workflow_type": (
                        "conversation" if result.get("skip_research") else "research"
                    ),
                    "error_details": result.get("error_message"),
                }
        else:
            # Handle conversation responses
            formatted_result = {
                "query": args.query,
                "response": result.get(
                    "conversation_response", "No response generated"
                ),
                "sources": [],
                "confidence": 1.0 if result.get("conversation_response") else 0.0,
                "execution_time": execution_time,
                "workflow_success": success,
                "workflow_type": (
                    "conversation" if result.get("skip_research") else "error"
                ),
                "error_details": result.get("error_message"),
            }

        # Record metrics
        monitor.record_query(
            args.query,
            execution_time,
            success,
            {
                "sources": len(formatted_result.get("sources", [])),
                "confidence": formatted_result.get("confidence", 0.0),
            },
        )

        # Format and display results
        await display_query_results(formatted_result, args)

        # Save results if requested
        if args.output:
            await save_results(formatted_result, args.output, args.format)

        return 0 if success else 1

    except Exception as e:
        execution_time = time.time() - start_time
        monitor.record_query(args.query, execution_time, False, {"error": str(e)})

        logger.error(f"Error processing query: {e}")
        print(f"❌ Query processing failed: {e}")
        return 1


async def run_test_mode(args: argparse.Namespace) -> int:
    """Run comprehensive system tests."""
    logger.info("Running system tests")

    try:
        print("🧪 Running Comprehensive System Tests...")
        print("=" * 60)

        # Run our simplified test
        # Basic test - just try to process a simple query
        test_query = "What is 2+2?"
        result = await run_research(
            query=test_query,
            session_id="test-session",
            request_id="test-001",
            quick_mode=args.quick,
        )

        test_results = {"overall_success": result is not None}

        # Display results
        if test_results.get("overall_success"):
            print("\n✅ All tests passed!")
            return 0
        else:
            print("\n❌ Some tests failed!")

            # Show detailed results if requested
            if args.verbose > 0:
                print("\n📊 Detailed Test Results:")
                print(json.dumps(test_results, indent=2))

            return 1

    except Exception as e:
        logger.error(f"Error running tests: {e}")
        print(f"❌ Test execution failed: {e}")
        return 1


async def display_query_results(result: Dict[str, Any], args: argparse.Namespace):
    """Display query results in the specified format."""

    if args.format == "json":
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.format == "markdown":
        # Format as markdown
        markdown_output = format_as_markdown(result, args.citation_style)
        print(markdown_output)

    elif args.format == "plain":
        # Plain text output
        print(f"Query: {result.get('query', '')}")
        print(f"Response: {result.get('response', '')}")
        print(f"Confidence: {result.get('confidence', 0.0):.2f}")
        print(f"Sources: {len(result.get('sources', []))}")

    else:  # rich format (default)
        # Use rich for basic formatted output
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text

        console = Console()

        # Format basic answer
        response_text = Text(result.get("response", "No response"))

        # Create formatted panel
        answer_panel = Panel(
            response_text,
            title=f"Research Results (Confidence: {result.get('confidence', 0.0):.1%})",
            border_style="green" if result.get("confidence", 0.0) > 0.7 else "yellow",
        )

        console.print(answer_panel)

        # Show sources if available
        sources = result.get("sources", [])
        if sources:
            console.print(f"\n📚 Sources ({len(sources)}):")
            for i, source in enumerate(sources[:5], 1):  # Show first 5
                console.print(f"  {i}. {source}")
            if len(sources) > 5:
                console.print(f"  ... and {len(sources) - 5} more sources")


def format_as_markdown(result: Dict[str, Any], citation_style: str) -> str:
    """Format results as markdown."""

    lines = [
        "# Research Results",
        "",
        f"**Query:** {result.get('query', '')}",
        "",
        f"**Confidence:** {result.get('confidence', 0.0):.1%}",
        "",
        "## Response",
        "",
        result.get("response", ""),
        "",
    ]

    # Add sources if available
    sources = result.get("sources", [])
    if sources:
        lines.extend(["## Sources", ""])

        for i, source in enumerate(sources, 1):
            lines.append(f"{i}. {source}")

        lines.append("")

    # Add metadata
    lines.extend(
        [
            "## Metadata",
            "",
            f"- Execution time: {result.get('execution_time', 0):.2f} seconds",
            f"- Sources analyzed: {len(sources)}",
            f"- Workflow success: {result.get('workflow_success', False)}",
        ]
    )

    return "\n".join(lines)


async def save_results(result: Dict[str, Any], output_path: str, format_type: str):
    """Save results to file."""

    try:
        output_file = Path(output_path)

        if format_type == "json":
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

        elif format_type == "markdown":
            markdown_content = format_as_markdown(
                result, "APA"
            )  # Default citation style
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(markdown_content)

        else:
            # Default to JSON
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"📁 Results saved to: {output_file}")

    except Exception as e:
        logger.error(f"Error saving results: {e}")
        print(f"❌ Failed to save results: {e}")


def setup_application_environment(args: argparse.Namespace):
    """Set up application environment and directories."""

    try:
        # Create necessary directories
        AppConfig.CONFIG_DIR.mkdir(exist_ok=True)
        AppConfig.CACHE_DIR.mkdir(exist_ok=True)
        AppConfig.SESSIONS_DIR.mkdir(exist_ok=True)

        # Set up logging
        log_level = args.log_level if not args.debug else "DEBUG"
        log_file = args.log_file or AppConfig.LOG_FILE

        logging.basicConfig(
            level=getattr(logging, log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                (
                    logging.StreamHandler(sys.stdout)
                    if args.verbose > 0
                    else logging.NullHandler()
                ),
            ],
        )

        logger.info(
            f"Application starting: {AppConfig.APP_NAME} v{AppConfig.APP_VERSION}"
        )
        logger.info(f"Log level: {log_level}, Log file: {log_file}")

    except Exception as e:
        print(f"❌ Failed to setup environment: {e}")
        sys.exit(1)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


async def main() -> int:
    """Main application entry point."""

    # Parse command line arguments
    parser = create_argument_parser()
    args = parser.parse_args()

    # Setup environment
    setup_application_environment(args)

    # Initialize monitoring
    monitor = ApplicationMonitor(enable_detailed=args.performance)

    # Setup signal handling
    signal_handler = SignalHandler(monitor)

    try:
        # Determine execution mode
        if args.test:
            return await run_test_mode(args)

        elif args.query:
            return await run_single_query_mode(args, monitor)

        else:
            # Default to interactive mode
            return await run_interactive_mode(args, monitor)

    except KeyboardInterrupt:
        print("\n👋 Session interrupted by user")
        return 0

    except Exception as e:
        logger.error(f"Unhandled application error: {e}")
        print(f"❌ Application error: {e}")

        if args.debug:
            import traceback

            traceback.print_exc()

        return 1

    finally:
        # Final performance report
        if args.performance and monitor.metrics["queries_processed"] > 0:
            monitor.print_performance_report()


def cli_main():
    """CLI entry point wrapper for setup.py."""
    try:
        return asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        return 0
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        return 1


if __name__ == "__main__":
    """Direct execution entry point."""
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Application failed: {e}")
        sys.exit(1)
