"""Exceptions for invalid experiment settings."""

class TNQSError(Exception):
    """Base exception for the numerical package."""

class InvalidExperiment(TNQSError, ValueError):
    """Physical or architectural choices do not match the scope."""
