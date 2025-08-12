"""
Output Formatting for AI Deep Research Assistant.

This module provides specialized formatters for academic citations, confidence visualization,
research trace display, and structured answer presentation that meet academic and
professional standards.
"""

import re
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
from urllib.parse import urlparse

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree

from ai_deep_research_assistant.graph.state.schemas import Citation
from ai_deep_research_assistant.agents.researcher import ResearchFinding

# =============================================================================
# FORMATTING CONFIGURATION
# =============================================================================


class FormatterConfig:
    """Configuration for output formatting styles and standards."""

    # Citation formats
    DEFAULT_CITATION_STYLE = "APA"
    SUPPORTED_CITATION_STYLES = ["APA", "MLA", "Chicago", "IEEE"]

    # Confidence visualization
    CONFIDENCE_BAR_WIDTH = 20
    CONFIDENCE_COLORS = {
        "high": "green",
        "medium": "yellow",
        "low": "red",
        "very_low": "bright_red",
    }

    # Answer section hierarchy
    SECTION_HIERARCHY = [
        "Executive Summary",
        "Key Findings",
        "Detailed Analysis",
        "Evidence",
        "Conclusions",
        "Limitations",
    ]

    # Visual elements
    PRIMARY_COLOR = "cyan"
    SECONDARY_COLOR = "blue"
    ACCENT_COLOR = "magenta"
    MUTED_COLOR = "dim"

    # Academic formatting
    MAX_TITLE_LENGTH = 80
    MAX_ABSTRACT_LENGTH = 200
    INDENT_SIZE = 2


# =============================================================================
# CITATION FORMATTING
# =============================================================================


class CitationFormatter:
    """
    Professional citation formatter supporting multiple academic standards.

    Provides formatting for APA, MLA, Chicago, and IEEE citation styles
    with proper handling of web sources, academic papers, and news articles.
    """

    def __init__(self, style: str = FormatterConfig.DEFAULT_CITATION_STYLE):
        """
        Initialize citation formatter.

        Args:
            style: Citation style (APA, MLA, Chicago, IEEE)
        """
        self.style = style.upper()
        if self.style not in FormatterConfig.SUPPORTED_CITATION_STYLES:
            self.style = FormatterConfig.DEFAULT_CITATION_STYLE

    def format_citation(self, citation: Union[Citation, Dict[str, Any]]) -> str:
        """
        Format a single citation according to the selected style.

        Args:
            citation: Citation object or dictionary

        Returns:
            Formatted citation string
        """
        try:
            # Extract citation data
            if isinstance(citation, dict):
                url = citation.get("url", "")
                title = citation.get("title", "Untitled")
                snippet = citation.get("snippet", "")
                domain = citation.get("domain", "")
                published_date = citation.get("published_date", "")
            else:
                url = getattr(citation, "url", "")
                title = getattr(citation, "title", "Untitled")
                snippet = getattr(citation, "snippet", "")
                domain = self._extract_domain(url)
                published_date = getattr(citation, "published_date", "")

            # Parse domain and determine source type
            source_info = self._analyze_source(url, domain, title)

            # Format according to style
            if self.style == "APA":
                return self._format_apa(title, url, domain, published_date, source_info)
            elif self.style == "MLA":
                return self._format_mla(title, url, domain, published_date, source_info)
            elif self.style == "Chicago":
                return self._format_chicago(
                    title, url, domain, published_date, source_info
                )
            elif self.style == "IEEE":
                return self._format_ieee(
                    title, url, domain, published_date, source_info
                )
            else:
                return self._format_apa(title, url, domain, published_date, source_info)

        except Exception as e:
            return f"[Citation formatting error: {str(e)}]"

    def format_bibliography(
        self, citations: List[Union[Citation, Dict[str, Any]]]
    ) -> str:
        """
        Format a complete bibliography with proper numbering and organization.

        Args:
            citations: List of citations to format

        Returns:
            Formatted bibliography string
        """
        try:
            if not citations:
                return "No sources to cite."

            formatted_citations = []

            # Sort citations by title for consistency
            sorted_citations = sorted(
                citations,
                key=lambda c: (
                    c.get("title", "")
                    if isinstance(c, dict)
                    else getattr(c, "title", "")
                ),
            )

            for i, citation in enumerate(sorted_citations, 1):
                formatted = self.format_citation(citation)

                if self.style in ["APA", "Chicago"]:
                    # Hanging indent style
                    formatted_citations.append(f"{formatted}")
                else:
                    # Numbered style
                    formatted_citations.append(f"{i:2d}. {formatted}")

            return "\n\n".join(formatted_citations)

        except Exception as e:
            return f"Bibliography formatting error: {str(e)}"

    def _format_apa(
        self, title: str, url: str, domain: str, date: str, source_info: Dict
    ) -> str:
        """Format citation in APA style."""
        parts = []

        # Author (use domain as organization if no author)
        if source_info["is_academic"]:
            parts.append(f"{domain} (Author).")
        else:
            parts.append(f"{domain}.")

        # Date
        if date:
            date_part = self._parse_date(date)
            parts.append(f"({date_part}).")
        else:
            parts.append("(n.d.).")

        # Title (italicized for web sources)
        clean_title = self._clean_title(title)
        parts.append(f"*{clean_title}*.")

        # URL and access date
        access_date = datetime.now().strftime("%B %d, %Y")
        parts.append(f"Retrieved {access_date}, from {url}")

        return " ".join(parts)

    def _format_mla(
        self, title: str, url: str, domain: str, date: str, source_info: Dict
    ) -> str:
        """Format citation in MLA style."""
        parts = []

        # Title in quotes
        clean_title = self._clean_title(title)
        parts.append(f'"{clean_title}."')

        # Website/organization name
        parts.append(f"*{domain}*,")

        # Date
        if date:
            date_part = self._parse_date(date, mla_format=True)
            parts.append(f"{date_part},")

        # URL
        parts.append(f"{url}.")

        # Access date
        access_date = datetime.now().strftime("%d %b %Y")
        parts.append(f"Accessed {access_date}.")

        return " ".join(parts)

    def _format_chicago(
        self, title: str, url: str, domain: str, date: str, source_info: Dict
    ) -> str:
        """Format citation in Chicago style."""
        parts = []

        # Organization/Website
        parts.append(f"{domain}.")

        # Title in quotes
        clean_title = self._clean_title(title)
        parts.append(f'"{clean_title}."')

        # Date
        if date:
            date_part = self._parse_date(date)
            parts.append(f"Last modified {date_part}.")

        # URL and access date
        access_date = datetime.now().strftime("%B %d, %Y")
        parts.append(f"{url} (accessed {access_date}).")

        return " ".join(parts)

    def _format_ieee(
        self, title: str, url: str, domain: str, date: str, source_info: Dict
    ) -> str:
        """Format citation in IEEE style."""
        parts = []

        # Author/Organization
        parts.append(f"{domain},")

        # Title in quotes
        clean_title = self._clean_title(title)
        parts.append(f'"{clean_title},"')

        # Date
        if date:
            date_part = self._parse_date(date, ieee_format=True)
            parts.append(f"{date_part}.")

        # URL and access information
        access_date = datetime.now().strftime("%b. %d, %Y")
        parts.append(f"[Online]. Available: {url}. [Accessed: {access_date}].")

        return " ".join(parts)

    def _extract_domain(self, url: str) -> str:
        """Extract clean domain name from URL."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            # Remove www prefix
            if domain.startswith("www."):
                domain = domain[4:]

            # Capitalize first letter
            return domain.capitalize()
        except:
            return "Unknown Source"

    def _analyze_source(self, url: str, domain: str, title: str) -> Dict[str, Any]:
        """Analyze source characteristics for appropriate formatting."""
        analysis = {
            "is_academic": False,
            "is_news": False,
            "is_government": False,
            "is_organization": False,
            "source_type": "web",
        }

        url_lower = url.lower()
        domain_lower = domain.lower()

        # Academic sources
        if any(
            indicator in url_lower
            for indicator in [".edu", "arxiv", "pubmed", "scholar", "jstor"]
        ):
            analysis["is_academic"] = True
            analysis["source_type"] = "academic"

        # Government sources
        elif ".gov" in url_lower:
            analysis["is_government"] = True
            analysis["source_type"] = "government"

        # News sources
        elif any(
            news in domain_lower for news in ["news", "cnn", "bbc", "reuters", "ap.org"]
        ):
            analysis["is_news"] = True
            analysis["source_type"] = "news"

        # Organization sources
        elif ".org" in url_lower:
            analysis["is_organization"] = True
            analysis["source_type"] = "organization"

        return analysis

    def _clean_title(self, title: str) -> str:
        """Clean and format title for citation."""
        if not title:
            return "Untitled"

        # Remove excessive whitespace
        title = re.sub(r"\s+", " ", title.strip())

        # Truncate if too long
        if len(title) > FormatterConfig.MAX_TITLE_LENGTH:
            title = title[: FormatterConfig.MAX_TITLE_LENGTH - 3] + "..."

        # Capitalize properly (title case)
        return title.title()

    def _parse_date(
        self, date_str: str, mla_format: bool = False, ieee_format: bool = False
    ) -> str:
        """Parse and format date string for citation."""
        if not date_str:
            return ""

        try:
            # Handle common date formats
            if "ago" in date_str.lower():
                # Relative dates like "2 hours ago"
                return datetime.now().strftime(
                    "%Y, %B %d" if not mla_format else "%d %B %Y"
                )

            # Return as-is for now (could be enhanced with more parsing)
            if mla_format:
                return date_str
            elif ieee_format:
                return date_str
            else:
                return date_str

        except:
            return date_str


# =============================================================================
# CONFIDENCE SCORE VISUALIZATION
# =============================================================================


class ConfidenceVisualizer:
    """
    Advanced confidence score visualization with multiple display modes.

    Provides visual indicators, progress bars, descriptive labels,
    and color coding for confidence levels.
    """

    def __init__(self, console: Optional[Console] = None):
        """
        Initialize confidence visualizer.

        Args:
            console: Rich Console instance for rendering
        """
        self.console = console or Console()

    def create_confidence_bar(self, confidence: float, width: int = None) -> Text:
        """
        Create a visual confidence bar with color coding.

        Args:
            confidence: Confidence score (0.0-1.0)
            width: Width of the bar in characters

        Returns:
            Rich Text object with formatted confidence bar
        """
        width = width or FormatterConfig.CONFIDENCE_BAR_WIDTH
        confidence = max(0.0, min(1.0, confidence))  # Clamp to valid range

        # Calculate filled portion
        filled_width = int(confidence * width)
        empty_width = width - filled_width

        # Determine color based on confidence level
        color = self._get_confidence_color(confidence)

        # Create bar components
        filled_part = "█" * filled_width
        empty_part = "░" * empty_width

        # Combine with color
        bar_text = Text()
        bar_text.append(filled_part, style=color)
        bar_text.append(empty_part, style="dim")

        return bar_text

    def create_confidence_indicator(
        self, confidence: float, include_percentage: bool = True
    ) -> Text:
        """
        Create a compact confidence indicator with icon and text.

        Args:
            confidence: Confidence score (0.0-1.0)
            include_percentage: Whether to include percentage value

        Returns:
            Rich Text object with confidence indicator
        """
        confidence = max(0.0, min(1.0, confidence))

        # Get confidence level and color
        level, icon = self._get_confidence_level(confidence)
        color = self._get_confidence_color(confidence)

        # Create indicator text
        indicator = Text()
        indicator.append(f"{icon} ", style=color)
        indicator.append(f"{level.title()}", style=color)

        if include_percentage:
            indicator.append(f" ({confidence:.1%})", style="dim")

        return indicator

    def create_confidence_panel(
        self,
        confidence: float,
        title: str = "Confidence Assessment",
        include_description: bool = True,
    ) -> Panel:
        """
        Create a detailed confidence panel with bar, level, and description.

        Args:
            confidence: Confidence score (0.0-1.0)
            title: Panel title
            include_description: Whether to include descriptive text

        Returns:
            Rich Panel with comprehensive confidence display
        """
        confidence = max(0.0, min(1.0, confidence))

        # Create components
        bar = self.create_confidence_bar(confidence)
        indicator = self.create_confidence_indicator(confidence)

        # Build panel content
        content_parts = [
            f"Score: {confidence:.3f} ({confidence:.1%})",
            "",
        ]

        # Add bar
        bar_line = Text("Bar:   ")
        bar_line.append(bar)
        content_parts.append(bar_line)

        # Add level indicator
        level_line = Text("Level: ")
        level_line.append(indicator)
        content_parts.append(level_line)

        # Add description if requested
        if include_description:
            description = self._get_confidence_description(confidence)
            content_parts.extend(["", f"Assessment: {description}"])

        # Determine panel color
        color = self._get_confidence_color(confidence)

        return Panel(
            "\n".join([str(part) for part in content_parts]),
            title=f"[bold {color}]{title}[/bold {color}]",
            border_style=color,
            padding=(1, 2),
        )

    def create_confidence_table(self, confidence_data: List[Dict[str, Any]]) -> Table:
        """
        Create a table displaying multiple confidence scores.

        Args:
            confidence_data: List of dicts with confidence info

        Returns:
            Rich Table with confidence comparisons
        """
        table = Table(
            title="Confidence Analysis", show_header=True, header_style="bold cyan"
        )
        table.add_column("Item", min_width=20)
        table.add_column("Score", width=8, justify="center")
        table.add_column("Level", width=12, justify="center")
        table.add_column("Visualization", min_width=25)

        for item_data in confidence_data:
            name = item_data.get("name", "Unknown")
            confidence = item_data.get("confidence", 0.0)

            # Create components
            score_text = f"{confidence:.3f}"
            level_indicator = self.create_confidence_indicator(
                confidence, include_percentage=False
            )
            bar = self.create_confidence_bar(confidence, width=15)

            table.add_row(name, score_text, level_indicator, bar)

        return table

    def _get_confidence_level(self, confidence: float) -> Tuple[str, str]:
        """Get confidence level and corresponding icon."""
        if confidence >= 0.8:
            return "high", "🟢"
        elif confidence >= 0.6:
            return "medium", "🟡"
        elif confidence >= 0.4:
            return "low", "🟠"
        else:
            return "very low", "🔴"

    def _get_confidence_color(self, confidence: float) -> str:
        """Get Rich color name for confidence level."""
        if confidence >= 0.8:
            return FormatterConfig.CONFIDENCE_COLORS["high"]
        elif confidence >= 0.6:
            return FormatterConfig.CONFIDENCE_COLORS["medium"]
        elif confidence >= 0.4:
            return FormatterConfig.CONFIDENCE_COLORS["low"]
        else:
            return FormatterConfig.CONFIDENCE_COLORS["very_low"]

    def _get_confidence_description(self, confidence: float) -> str:
        """Get descriptive text for confidence level."""
        if confidence >= 0.9:
            return (
                "Very high confidence - multiple reliable sources in strong agreement"
            )
        elif confidence >= 0.8:
            return "High confidence - well-supported by credible sources"
        elif confidence >= 0.7:
            return "Good confidence - supported by reliable sources with minor gaps"
        elif confidence >= 0.6:
            return "Moderate confidence - reasonable support but some uncertainty"
        elif confidence >= 0.5:
            return "Fair confidence - limited sources or conflicting information"
        elif confidence >= 0.3:
            return "Low confidence - insufficient or unreliable sources"
        else:
            return "Very low confidence - minimal or poor quality sources"


# =============================================================================
# RESEARCH TRACE DISPLAY
# =============================================================================


class ResearchTraceFormatter:
    """
    Formats research reasoning traces and decision processes.

    Displays step-by-step reasoning, agent decisions, source evaluation,
    and confidence assessments in a clear, hierarchical format.
    """

    def __init__(self, console: Optional[Console] = None):
        """
        Initialize research trace formatter.

        Args:
            console: Rich Console instance
        """
        self.console = console or Console()

    def format_research_trace(
        self,
        reasoning_steps: List[Dict[str, Any]],
        title: str = "Research Reasoning Trace",
    ) -> Panel:
        """
        Format a complete research reasoning trace.

        Args:
            reasoning_steps: List of reasoning step dictionaries
            title: Title for the trace display

        Returns:
            Rich Panel with formatted research trace
        """
        if not reasoning_steps:
            return Panel(
                "[dim]No reasoning trace available[/dim]",
                title=f"[bold {FormatterConfig.MUTED_COLOR}]{title}[/bold {FormatterConfig.MUTED_COLOR}]",
                border_style=FormatterConfig.MUTED_COLOR,
            )

        # Create tree structure for reasoning
        trace_tree = Tree(
            f"[bold {FormatterConfig.PRIMARY_COLOR}]Research Process[/bold {FormatterConfig.PRIMARY_COLOR}]"
        )

        for i, step in enumerate(reasoning_steps, 1):
            step_description = step.get("description", f"Step {i}")
            step_process = step.get("process", "Process details not available")
            step_outcome = step.get("outcome", "Outcome not specified")

            # Add step to tree
            step_node = trace_tree.add(f"[bold]{step_description}[/bold]")
            step_node.add(f"[cyan]Process:[/cyan] {step_process}")
            step_node.add(f"[green]Outcome:[/green] {step_outcome}")

        return Panel(
            trace_tree,
            title=f"[bold {FormatterConfig.PRIMARY_COLOR}]{title}[/bold {FormatterConfig.PRIMARY_COLOR}]",
            border_style=FormatterConfig.PRIMARY_COLOR,
            padding=(1, 2),
        )

    def format_decision_points(self, decisions: List[Dict[str, Any]]) -> Table:
        """
        Format key decision points in research process.

        Args:
            decisions: List of decision point dictionaries

        Returns:
            Rich Table with decision analysis
        """
        table = Table(
            title="Key Decision Points", show_header=True, header_style="bold magenta"
        )
        table.add_column("Step", width=6, justify="center")
        table.add_column("Decision", min_width=30)
        table.add_column("Rationale", min_width=40)
        table.add_column("Confidence", width=12, justify="center")

        for i, decision in enumerate(decisions, 1):
            decision_text = decision.get("decision", "Decision not specified")
            rationale = decision.get("rationale", "Rationale not provided")
            confidence = decision.get("confidence", 0.5)

            # Create confidence indicator
            visualizer = ConfidenceVisualizer()
            confidence_indicator = visualizer.create_confidence_indicator(confidence)

            table.add_row(str(i), decision_text, rationale, confidence_indicator)

        return table

    def format_source_evaluation(self, evaluations: List[Dict[str, Any]]) -> Table:
        """
        Format source evaluation and quality assessment.

        Args:
            evaluations: List of source evaluation dictionaries

        Returns:
            Rich Table with source analysis
        """
        table = Table(
            title="Source Quality Evaluation",
            show_header=True,
            header_style="bold blue",
        )
        table.add_column("Source", min_width=25)
        table.add_column("Type", width=12)
        table.add_column("Credibility", width=15, justify="center")
        table.add_column("Relevance", width=15, justify="center")
        table.add_column("Assessment", min_width=30)

        for evaluation in evaluations:
            source_name = evaluation.get("source", "Unknown Source")
            source_type = evaluation.get("type", "Web")
            credibility = evaluation.get("credibility", 0.5)
            relevance = evaluation.get("relevance", 0.5)
            assessment = evaluation.get("assessment", "No assessment available")

            # Create visual indicators
            visualizer = ConfidenceVisualizer()
            credibility_bar = visualizer.create_confidence_bar(credibility, width=8)
            relevance_bar = visualizer.create_confidence_bar(relevance, width=8)

            table.add_row(
                source_name, source_type, credibility_bar, relevance_bar, assessment
            )

        return table


# =============================================================================
# ANSWER SECTION FORMATTING
# =============================================================================


class AnswerFormatter:
    """
    Formats research answers into structured, professional sections.

    Provides hierarchical organization, proper headings, citation integration,
    and academic-standard presentation of research results.
    """

    def __init__(self, citation_style: str = "APA"):
        """
        Initialize answer formatter.

        Args:
            citation_style: Citation style for references
        """
        self.citation_formatter = CitationFormatter(citation_style)

    def format_complete_answer(
        self,
        query: str,
        response: str,
        findings: List[ResearchFinding],
        citations: List[Citation],
        confidence: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Panel:
        """
        Format a complete research answer with all sections.

        Args:
            query: Original research query
            response: Main response text
            findings: Research findings list
            citations: Citation list
            confidence: Overall confidence score
            metadata: Additional metadata

        Returns:
            Rich Panel with complete formatted answer
        """
        sections = []

        # Executive Summary
        executive_summary = self._create_executive_summary(
            query, response, confidence, metadata
        )
        sections.append(executive_summary)

        # Main Response
        main_response = self._format_main_response(response)
        sections.append(main_response)

        # Key Findings
        if findings:
            key_findings = self._format_key_findings(findings)
            sections.append(key_findings)

        # Citations
        if citations:
            citations_section = self._format_citations_section(citations)
            sections.append(citations_section)

        # Research Metadata
        if metadata:
            metadata_section = self._format_metadata_section(metadata)
            sections.append(metadata_section)

        # Combine all sections
        complete_content = "\n\n".join(sections)

        # Determine overall panel styling
        color = self._get_answer_color(confidence)

        return Panel(
            complete_content,
            title=f"[bold {color}]Research Analysis[/bold {color}]",
            border_style=color,
            padding=(1, 2),
        )

    def _create_executive_summary(
        self,
        query: str,
        response: str,
        confidence: float,
        metadata: Dict[str, Any] = None,
    ) -> str:
        """Create executive summary section."""
        # Check for rate limit notice in metadata
        rate_limit_notice = ""
        if metadata and metadata.get("rate_limit_notice"):
            rate_limit_notice = f"\n\n⚠️ **Notice:** {metadata['rate_limit_notice']}"

        # Extract first paragraph or first 200 characters as summary
        if len(response) > FormatterConfig.MAX_ABSTRACT_LENGTH:
            summary = (
                response[: FormatterConfig.MAX_ABSTRACT_LENGTH].rsplit(".", 1)[0] + "."
            )
        else:
            summary = response.split("\n\n")[0] if "\n\n" in response else response

        # Add confidence indicator
        visualizer = ConfidenceVisualizer()
        confidence_text = visualizer.create_confidence_indicator(confidence)

        return f"""## Executive Summary

**Query:** {query}
{rate_limit_notice}
**Summary:** {summary}

**Overall Confidence:** {confidence_text}"""

    def _format_main_response(self, response: str) -> str:
        """Format the main response section."""
        return f"""## Analysis

{response}"""

    def _format_key_findings(self, findings: List[ResearchFinding]) -> str:
        """Format key findings section."""
        findings_text = ["## Key Findings\n"]

        for i, finding in enumerate(findings[:5], 1):  # Limit to top 5 findings
            claim = finding.claim if hasattr(finding, "claim") else str(finding)
            confidence = getattr(finding, "confidence", 0.0)

            # Create confidence indicator
            visualizer = ConfidenceVisualizer()
            confidence_indicator = visualizer.create_confidence_indicator(confidence)

            findings_text.append(f"**{i}.** {claim} {confidence_indicator}")

        return "\n\n".join(findings_text)

    def _format_citations_section(self, citations: List[Citation]) -> str:
        """Format citations section."""
        bibliography = self.citation_formatter.format_bibliography(citations)

        return f"""## References

{bibliography}"""

    def _format_metadata_section(self, metadata: Dict[str, Any]) -> str:
        """Format research metadata section."""
        metadata_lines = ["## Research Metadata\n"]

        # Add relevant metadata
        if metadata.get("execution_time"):
            metadata_lines.append(
                f"**Execution Time:** {metadata['execution_time']:.2f} seconds"
            )

        if metadata.get("sources_analyzed"):
            metadata_lines.append(
                f"**Sources Analyzed:** {metadata['sources_analyzed']}"
            )

        if metadata.get("research_method"):
            metadata_lines.append(f"**Research Method:** {metadata['research_method']}")

        if metadata.get("agents_used"):
            agents = ", ".join(metadata["agents_used"])
            metadata_lines.append(f"**Research Agents:** {agents}")

        return "\n".join(metadata_lines)

    def _get_answer_color(self, confidence: float) -> str:
        """Get color for answer panel based on confidence."""
        if confidence >= 0.8:
            return "green"
        elif confidence >= 0.6:
            return "yellow"
        else:
            return "red"


# =============================================================================
# TESTING AND UTILITIES
# =============================================================================


def test_formatters():
    """Test all formatter components."""
    print("🧪 Testing output formatters...")

    console = Console()

    try:
        # Test Citation Formatter
        print("📚 Testing citation formatter...")
        citation_formatter = CitationFormatter("APA")

        test_citation = {
            "url": "https://example.com/article",
            "title": "Test Article About Machine Learning",
            "domain": "example.com",
            "published_date": "2024-01-15",
        }

        formatted_citation = citation_formatter.format_citation(test_citation)
        print(f"✅ Citation formatted: {formatted_citation[:80]}...")

        # Test Confidence Visualizer
        print("📊 Testing confidence visualizer...")
        confidence_viz = ConfidenceVisualizer(console)

        confidence_bar = confidence_viz.create_confidence_bar(0.85)
        confidence_indicator = confidence_viz.create_confidence_indicator(0.85)

        console.print("✅ Confidence bar:", confidence_bar)
        console.print("✅ Confidence indicator:", confidence_indicator)

        # Test Research Trace Formatter
        print("🔍 Testing research trace formatter...")
        trace_formatter = ResearchTraceFormatter(console)

        test_steps = [
            {
                "description": "Query Analysis",
                "process": "Analyzed user query for research requirements",
                "outcome": "Identified as research request requiring multiple sources",
            },
            {
                "description": "Source Search",
                "process": "Searched academic and web sources",
                "outcome": "Found 5 relevant sources with high credibility",
            },
        ]

        trace_panel = trace_formatter.format_research_trace(test_steps)
        console.print(trace_panel)

        # Test Answer Formatter
        print("📝 Testing answer formatter...")
        answer_formatter = AnswerFormatter("APA")

        test_response = "Machine learning is a subset of artificial intelligence that enables systems to learn from data."

        complete_answer = answer_formatter.format_complete_answer(
            query="What is machine learning?",
            response=test_response,
            findings=[],
            citations=[test_citation],
            confidence=0.85,
        )

        console.print(complete_answer)

        print("🎉 All formatter tests completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Formatter testing failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    """Test formatters when run directly"""
    success = test_formatters()
    exit(0 if success else 1)
