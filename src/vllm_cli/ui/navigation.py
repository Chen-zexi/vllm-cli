#!/usr/bin/env python3
"""
Unified navigation system for vLLM CLI.

Provides consistent menu navigation and user interaction.
"""
import logging
from typing import List, Optional

import inquirer
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()


def unified_prompt(
    key: str, message: str, choices: List[str], allow_back: bool = True
) -> Optional[str]:
    """
    Display a unified prompt for user selection.

    Args:
        key: Unique key for this prompt
        message: Prompt message to display
        choices: List of choices
        allow_back: Whether to include a Back option

    Returns:
        Selected choice or None if cancelled
    """
    try:
        # Add navigation options
        prompt_choices = choices.copy()
        if allow_back:
            prompt_choices.append("← Back")

        # Create inquirer prompt
        questions = [
            inquirer.List(
                key,
                message=message,
                choices=prompt_choices,
            )
        ]

        # Get user selection
        answers = inquirer.prompt(questions)

        if not answers:
            return None

        answer = answers[key]

        # Handle navigation
        if answer == "← Back":
            return "BACK"

        return answer

    except KeyboardInterrupt:
        logger.info("User interrupted prompt")
        return None
    except Exception as e:
        logger.error(f"Error in unified prompt: {e}")
        return None
