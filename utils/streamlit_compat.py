"""Streamlit compatibility shims for older versions."""

import streamlit as st


def setup_streamlit_compatibility():
    """Setup compatibility shims for newer Streamlit APIs that may not exist in older versions."""
    try:
        _st = st
    except Exception:
        _st = None

    if _st is not None:
        # toggle -> fallback to checkbox
        if not hasattr(st, 'toggle'):
            def _toggle(label, value=False, **kwargs):
                return st.checkbox(label, value)
            setattr(st, 'toggle', _toggle)

        # status -> fallback to a dummy context manager with an update method
        if not hasattr(st, 'status'):
            class _DummyStatus:
                def __init__(self, *a, **k):
                    pass
                def __enter__(self):
                    return self
                def __exit__(self, exc_type, exc, tb):
                    return False
                def update(self, label=None, state=None):
                    # best-effort: show a spinner or simple text
                    try:
                        if label:
                            st.write(label)
                    except Exception:
                        pass

            def _status(msg, expanded=False):
                return _DummyStatus()

            setattr(st, 'status', _status)

        # data_editor -> fallback to dataframe display and return the passed DataFrame
        if not hasattr(st, 'data_editor'):
            def _data_editor(df, **kwargs):
                try:
                    st.dataframe(df)
                except Exception:
                    pass
                return df
            setattr(st, 'data_editor', _data_editor)

        # divider -> fallback to markdown horizontal rule
        if not hasattr(st, 'divider'):
            def _divider():
                try:
                    st.markdown('---')
                except Exception:
                    pass
            setattr(st, 'divider', _divider)

        # fragment decorator -> no-op decorator when missing
        if not hasattr(st, 'fragment'):
            def _fragment(fn):
                return fn
            setattr(st, 'fragment', _fragment)

        # Provide a minimal column_config namespace to avoid attribute errors when building
        # column_config objects; these dummies are ignored by our fallback data_editor.
        if not hasattr(st, 'column_config'):
            class _DummyCol:
                def __init__(self, *a, **k):
                    pass

            class _DummyColConfig:
                TextColumn = _DummyCol
                SelectboxColumn = _DummyCol
                NumberColumn = _DummyCol
                DatetimeColumn = _DummyCol

            setattr(st, 'column_config', _DummyColConfig())
