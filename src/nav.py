"""Registry of navigation pages so views can link to each other with st.page_link."""

PAGES: dict = {}


def register(name, page):
    PAGES[name] = page
    return page


def get(name):
    return PAGES.get(name)
