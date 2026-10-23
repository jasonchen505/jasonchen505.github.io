# Adding a blog post

Create `_posts/YYYY-MM-DD-short-title.md` with a date that is not in the future:

```markdown
---
title: "Your article title"
---

Write the article here in Markdown.
```

The post uses the shared post layout, appears automatically in `/blog/`, and has its own URL `/blog/short-title/`. Do not repeat the title as a heading in the article body.

For a private draft, add `published: false` to its front matter. No sample posts are included in the site.
