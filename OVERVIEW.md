# What This Project Does (Plain-English Version)

*The [README](README.md) is written for other engineers. This page is for
anyone who wants the gist without the statistics vocabulary.*

## The problem

Every trading day, 20 different countries' currencies move against each
other: the Euro, the Japanese Yen, the Mexican Peso, and so on. That's 20
numbers changing every day, for about 13 years of data. Staring at 20 lines
on a chart at once tells you almost nothing; they're too tangled together to
see what's actually going on.

## What this does

**Step 1: Find the hidden storylines.** Instead of tracking all 20
currencies individually, the code looks for a small number of underlying
"storylines" that explain most of what's happening. Think of it like
summarizing a symphony by naming its three main melodies instead of
transcribing every instrument. Here, three storylines turn out to explain
about 70%, 15%, and 15% of the action.

**Step 2: Forecast how bumpy each storyline will be.** Financial markets
go through calm stretches and turbulent ones ("volatility clustering,"
meaning a calm day tends to follow a calm day, a wild day tends to follow a
wild day). The code fits a family of models that are specifically built to
forecast *how much things will bounce around* in the near future, not just
where they'll end up, similar to how a weather forecast gives you a range
of likely temperatures, not one number.

## What it found

Without being told anything about geography or economics, the analysis
automatically split the 20 currencies into two groups: established,
developed-market currencies (Euro, British Pound, Swiss Franc, Japanese Yen,
Canadian/Australian/New Zealand Dollars, and several Northern/Central
European ones) moving one way, and emerging-market currencies (Turkish Lira,
Mexican Peso especially) moving the opposite way. That's a real,
economically sensible pattern the math discovered on its own. Nobody
pre-labeled the currencies as "developed" or "emerging" beforehand.

## Why this project, specifically

This started as a graduate statistics course project (Time Series Analysis,
using MATLAB, common in academic research but rare in industry). What's in
this repo is a from-scratch rebuild in Python, done independently rather
than by directly translating the original code line-by-line, for a
practical reason: the original relied on a piece of academic research code
with a license that doesn't allow redistribution. Swapping in an equivalent,
properly-licensed approach and rebuilding the surrounding logic from the
published formulas, rather than just copying it, is exactly the kind of
judgment call that comes up when moving real code from a research or
legacy setting into something shareable or production-ready.

## What this demonstrates

- **Translating research code into something usable.** A lot of real data
  science work is exactly this: taking something that works in a notebook
  or a legacy tool and turning it into clean, documented, reusable code.
- **Choosing between competing models with a defensible method**, rather
  than eyeballing which one "looks right."
- **Catching your own mistakes.** Running the full pipeline end-to-end
  surfaced two real bugs (one made every result silently meaningless, the
  other made the reported numbers off by a hidden constant). Both are
  fixed, and both now have a permanent automated test guarding against them
  coming back.
- **Paying attention to where code and data come from**, including
  licensing, instead of just making something work.

See the [README](README.md) for the technical details, how to run it, and
the actual output.
