"""The MCP server is the openzoo front door -- it gets a CI hook like every other organ.
(Last-chance wiring sweep: holographic_mcp.py lives at top level, outside the buried-selftest
audit's holographic/** scope, so without this file the zoo's entire mount surface had zero CI
coverage. A front door nobody tests is a gap wearing a doorknob.)"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_mcp_selftest_in_process():
    from holographic_mcp import _selftest
    _selftest()          # initialize, tool list exact, corpus + memory + void round trips,
                         # partition persistence, private refusal, -32601 -- all pinned inside
