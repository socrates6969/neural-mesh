"""Shared method-card memory for the neural mesh (see train/memory/README in the module docstrings).

Import is deliberately light: the hook's warm path only needs the socket client (train.memory.client), never numpy.
Public API: client.lookup / client.observe (daemon with in-process fallback), api.retrieve / api.render / api.observe.
"""
