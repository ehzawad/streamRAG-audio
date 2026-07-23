# CRAG text evaluation dataset — human review sheet

> **Status: PENDING HUMAN REVIEW.** Do not freeze or run the unseen test split
> until every item and the global-corpus construction have been reviewed.

This is a dataset-quality review, not an application or model-output review. Work
only from each question, expected answer, source pages, and the proposed support
document. Do not inspect Naive RAG or StreamRAG predictions while deciding whether
an item belongs in the evaluation set.

## What the fields mean

- **Gold** is the expected answer; **Aliases** are equivalent scorer-accepted forms.
- **Own-page evidence covered** says whether a selected source page supports the
  answer, or supports the intended abstention for a false premise.
- **Proposed supporting document IDs** name the exact committed corpus pages to
  verify; they are candidates until this review is complete.
- **Candidate stabilization class** predicts when the information need becomes
  clear while a person types. It is a low-confidence review aid, not a result.
- **Role** is `dev` for visible development items and `test` for the sealed final
  comparison. Accepting it confirms only that split assignment.

## How to review

1. Find each proposed ID in `documents.jsonl.bz2` (for example, `bzgrep
   '<document-id>' data/crag_eval/documents.jsonl.bz2`) and verify wording, time anchor, expected
   answer, and aliases against that committed text and the listed source pages.
2. Read the query left to right and judge whether its stabilization class is
   plausible without viewing either path's outputs.
3. Confirm the development/test role, tick the six item boxes, and record any
   correction in the dataset change before approval.
4. Complete the corpus checklist at the end. Ticking boxes documents review; the
   approval status remains separate until all corrections are resolved.

## crag-text-dev-001 — dev

- **Words:** 15
- **Domain / type / dynamism:** finance / simple / static
- **Query:** how long does a stock need to be held to make capital gains long term?
- **Candidate stabilization class:** early_stabilization
- **Classification confidence:** low
- **Heuristic reason:** a non-generic lexical anchor appears in the first half without a late constraint
- **Gold:** more than one year
- **Aliases:** over one year, longer than one year
- **Own-page evidence covered:** yes
- **Proposed supporting document IDs:** crag-global-9d22ffbe3ca22f9a9858bd6e
- **Original CRAG pages:**
  - Differences of Short Term vs Long Term Capital Gains - SmartAsset ...: <https://smartasset.com/financial-advisor/short-term-vs-long-term-capital-gains>
  - Topic no. 409, Capital gains and losses | Internal Revenue Service: <https://www.irs.gov/taxtopics/tc409>
  - Taxes on Stocks: What You Have to Pay, How to Pay Less - NerdWallet: <https://www.nerdwallet.com/article/taxes/taxes-on-stocks>
  - Short-Term vs. Long-Term Capital Gains Taxes | Charles Schwab: <https://www.schwab.com/learn/story/short-term-vs-long-term-capital-gains-taxes>
- [ ] Question wording and time anchor are clear
- [ ] Expected answer and aliases are factually correct
- [ ] Committed support document proves the answer or intended abstention
- [ ] Typing can plausibly stabilize as classified
- [ ] Stabilization class accepted or corrected without model outputs
- [ ] Split assignment accepted (development or test)

## crag-text-dev-002 — dev

- **Words:** 9
- **Domain / type / dynamism:** movie / comparison / static
- **Query:** which dune movie has better music, 1984 or 2021?
- **Candidate stabilization class:** revision_or_ambiguity
- **Classification confidence:** low
- **Heuristic reason:** a late disjunction introduces another candidate after a plausible prefix
- **Gold:** Dune (2021)
- **Aliases:** the 2021 Dune, Dune 2021, the music for Dune 2021 is more striking
- **Own-page evidence covered:** yes
- **Proposed supporting document IDs:** crag-global-046532bde5668fd1929c2408
- **Original CRAG pages:**
  - Dune (2021) vs. Dune (1984): What Are the Differences? | Den of Geek: <https://www.denofgeek.com/movies/dune-2021-vs-dune-1984-the-differences/>
  - 27 Differences Between The New "Dune" Movie And The One From 1984: <https://www.buzzfeed.com/evelinamedina/dune-differences-between-2021-1984>
  - Dune: 4 Things The 1984 Movie Got Right (& 6 Things The 2021 Version ...: <https://screenrant.com/dune-1984-2021-comparison/>
  - r/dune on Reddit: Dune 1984 was better than Dune 2021: <https://www.reddit.com/r/dune/comments/q775he/dune_1984_was_better_than_dune_2021/>
- [ ] Question wording and time anchor are clear
- [ ] Expected answer and aliases are factually correct
- [ ] Committed support document proves the answer or intended abstention
- [ ] Typing can plausibly stabilize as classified
- [ ] Stabilization class accepted or corrected without model outputs
- [ ] Split assignment accepted (development or test)

## crag-text-dev-003 — dev

- **Words:** 19
- **Domain / type / dynamism:** music / multi-hop / static
- **Query:** what is the name of the bad bunny album released before nadie sabe lo que va a pasar mañana?
- **Candidate stabilization class:** early_stabilization
- **Classification confidence:** low
- **Heuristic reason:** a non-generic lexical anchor appears in the first half without a late constraint
- **Gold:** Un Verano Sin Ti
- **Aliases:** —
- **Own-page evidence covered:** yes
- **Proposed supporting document IDs:** crag-global-a6deebaec508edb9daefce6e
- **Original CRAG pages:**
  - Bad Bunny's new album is here: All the details about 'Nadie Sabe ...: <https://www.usatoday.com/story/entertainment/music/2023/10/13/bad-bunny-new-album-nadie-sabe-tracklist-features-translation/71167097007/>
  - Bad Bunny Confirms New Album 'Nadie Sabe Lo Que Va a Pasar Mañana' ...: <https://variety.com/2023/music/news/bad-bunny-new-album-nadie-sabe-lo-que-va-a-pasar-manana-1235749333/>
  - Nadie Sabe Lo Que Va a Pasar Mañana - Wikipedia: <https://en.wikipedia.org/wiki/Nadie_Sabe_Lo_Que_Va_a_Pasar_Ma%C3%B1ana>
- [ ] Question wording and time anchor are clear
- [ ] Expected answer and aliases are factually correct
- [ ] Committed support document proves the answer or intended abstention
- [ ] Typing can plausibly stabilize as classified
- [ ] Stabilization class accepted or corrected without model outputs
- [ ] Split assignment accepted (development or test)

## crag-text-dev-004 — dev

- **Words:** 13
- **Domain / type / dynamism:** open / false_premise / static
- **Query:** Was Taylor Swift's debut album Fearless, released in the United States in 2008?
- **Original source query:** did taylor swifts debut album fearless launched in 2008 in us?
- **Candidate stabilization class:** late_stabilization
- **Classification confidence:** low
- **Heuristic reason:** the first lexical anchor or a material constraint occurs late in the query
- **Gold:** invalid question
- **Aliases:** —
- **Own-page evidence covered:** yes
- **Proposed supporting document IDs:** crag-global-bd5b1cb86ab3f862b0ed8803
- **Original CRAG pages:**
  - Taylor Swift: Fearless Era (2008-2010) | Taylor Swift Switzerland: <https://taylorswiftswitzerland.ch/index.php/album-eras/fearless-era/>
  - Fearless (Taylor Swift album) - Wikipedia: <https://en.wikipedia.org/wiki/Fearless_(Taylor_Swift_album)>
  - Fearless | album by Swift | Britannica: <https://www.britannica.com/topic/Fearless-album-by-Swift>
  - Fearless | Taylor Swift Wiki | Fandom: <https://taylorswift.fandom.com/wiki/Fearless>
- [ ] Question wording and time anchor are clear
- [ ] Expected answer and aliases are factually correct
- [ ] Committed support document proves the answer or intended abstention
- [ ] Typing can plausibly stabilize as classified
- [ ] Stabilization class accepted or corrected without model outputs
- [ ] Split assignment accepted (development or test)

## crag-text-dev-005 — dev

- **Words:** 14
- **Domain / type / dynamism:** sports / simple_w_condition / fast-changing
- **Query:** who is currently ranked as the number one mens tennis player in the world?
- **Candidate stabilization class:** early_stabilization
- **Classification confidence:** low
- **Heuristic reason:** a non-generic lexical anchor appears in the first half without a late constraint
- **Gold:** Novak Djokovic
- **Aliases:** —
- **Own-page evidence covered:** yes
- **Proposed supporting document IDs:** crag-global-a4048eeaa02d759267b94091
- **Original CRAG pages:**
  - Players & Rankings | Tennis.com: <https://www.tennis.com/players-rankings/>
  - ATP rankings: best male tennis players 2024 | Statista: <https://www.statista.com/statistics/264896/atp-ranking-of-the-top-50-male-tennis-players-worldwide/>
  - Who is the world No.1 in men's tennis? Updated ATP rankings and ...: <https://www.sportingnews.com/us/tennis/news/world-no-1-tennis-men-atp-rankings-explainer-updated/w5kranveoicurzao0msj7fh1>
  - Men's Tennis Rankings - Tennis - ESPN: <https://www.espn.com/sports/tennis/rankings>
- [ ] Question wording and time anchor are clear
- [ ] Expected answer and aliases are factually correct
- [ ] Committed support document proves the answer or intended abstention
- [ ] Typing can plausibly stabilize as classified
- [ ] Stabilization class accepted or corrected without model outputs
- [ ] Split assignment accepted (development or test)

## crag-text-test-001 — test

- **Words:** 8
- **Domain / type / dynamism:** finance / multi-hop / slow-changing
- **Query:** where did the ceo of salesforce previously work?
- **Candidate stabilization class:** early_stabilization
- **Classification confidence:** low
- **Heuristic reason:** a non-generic lexical anchor appears in the first half without a late constraint
- **Gold:** Oracle
- **Aliases:** Oracle Corporation, 13 years at Oracle, Marc Benioff spent 13 years at Oracle, before launching Salesforce.
- **Own-page evidence covered:** yes
- **Proposed supporting document IDs:** crag-global-0f4db44459e874d9300f90a8
- **Original CRAG pages:**
  - The History of Salesforce - Salesforce: <https://www.salesforce.com/news/stories/the-history-of-salesforce/>
  - Marc Benioff - Wikipedia: <https://en.wikipedia.org/wiki/Marc_Benioff>
  - Leadership - Salesforce.com US: <https://www.salesforce.com/company/leadership/>
  - Salesforce - Wikipedia: <https://en.wikipedia.org/wiki/Salesforce>
- [ ] Question wording and time anchor are clear
- [ ] Expected answer and aliases are factually correct
- [ ] Committed support document proves the answer or intended abstention
- [ ] Typing can plausibly stabilize as classified
- [ ] Stabilization class accepted or corrected without model outputs
- [ ] Split assignment accepted (development or test)

## crag-text-test-002 — test

- **Words:** 9
- **Domain / type / dynamism:** finance / simple_w_condition / static
- **Query:** Who stepped down as Apple's CEO in August 2011?
- **Original source query:** who was the ceo of apple in 2010?
- **Candidate stabilization class:** late_stabilization
- **Classification confidence:** low
- **Heuristic reason:** the first lexical anchor or a material constraint occurs late in the query
- **Gold:** Steve Jobs
- **Aliases:** —
- **Own-page evidence covered:** yes
- **Proposed supporting document IDs:** crag-global-6e07afd4ce9ef9c61f0c659e
- **Original CRAG pages:**
  - Who Is Tim Cook?: <https://www.investopedia.com/tim-cook-5224249>
  - Tim Cook took the helm at Apple over 10 years ago. Here's how he ...: <https://www.businessinsider.com/the-rise-of-apple-ceo-tim-cook-2016-1>
  - Tim Cook, Apple CEO, Auburn University Commencement Speech 2010: <https://www.fastcompany.com/1776338/tim-cook-apple-ceo-auburn-university-commencement-speech-2010>
- [ ] Question wording and time anchor are clear
- [ ] Expected answer and aliases are factually correct
- [ ] Committed support document proves the answer or intended abstention
- [ ] Typing can plausibly stabilize as classified
- [ ] Stabilization class accepted or corrected without model outputs
- [ ] Split assignment accepted (development or test)

## crag-text-test-003 — test

- **Words:** 24
- **Domain / type / dynamism:** movie / comparison / static
- **Query:** Which film had the larger domestic opening weekend: Harry Potter and the Half-Blood Prince or Harry Potter and the Deathly Hallows – Part 2?
- **Original source query:** which film had the higher grossing weekend, harry potter and the half-blood prince or harry potter and the deathly hallows – part 2?
- **Candidate stabilization class:** revision_or_ambiguity
- **Classification confidence:** low
- **Heuristic reason:** a late disjunction introduces another candidate after a plausible prefix
- **Gold:** Harry Potter and the Deathly Hallows – Part 2
- **Aliases:** Harry Potter and the Deathly Hallows: Part 2, Deathly Hallows Part 2
- **Own-page evidence covered:** yes
- **Proposed supporting document IDs:** crag-global-14a3d0579cdfdede70eafa7e
- **Original CRAG pages:**
  - 6. Harry Potter and the Half-Blood Prince: <https://www.forbes.com/pictures/gkhf45ikm/6-harry-potter-and-the-h/amp/>
  - Every 'Harry Potter' And Wizarding World Box Office Opening Ranked ...: <https://www.forbes.com/sites/simonthompson/2018/11/18/every-harry-potter-and-wizarding-world-box-office-opening-ranked-worst-to-best/>
  - Harry Potter and the Deathly Hallows – Part 2 - Wikipedia: <https://en.wikipedia.org/wiki/Harry_Potter_and_the_Deathly_Hallows_–_Part_2>
  - Harry Potter movies: production costs and global box office revenue ...: <https://www.statista.com/statistics/323356/harry-potter-production-costs-box-office-revenue/>
  - All Harry Potter Movies, Ranked by How Expensive They Were to Make: <https://movieweb.com/harry-potter-movies-budget/>
- [ ] Question wording and time anchor are clear
- [ ] Expected answer and aliases are factually correct
- [ ] Committed support document proves the answer or intended abstention
- [ ] Typing can plausibly stabilize as classified
- [ ] Stabilization class accepted or corrected without model outputs
- [ ] Split assignment accepted (development or test)

## crag-text-test-004 — test

- **Words:** 12
- **Domain / type / dynamism:** movie / simple / slow-changing
- **Query:** which actress won an academy award for her role in "black swan"?
- **Candidate stabilization class:** early_stabilization
- **Classification confidence:** low
- **Heuristic reason:** a non-generic lexical anchor appears in the first half without a late constraint
- **Gold:** Natalie Portman
- **Aliases:** —
- **Own-page evidence covered:** yes
- **Proposed supporting document IDs:** crag-global-4ed5e8a8fb93221e56e3e977
- **Original CRAG pages:**
  - Black Swan (film) - Wikipedia: <https://en.wikipedia.org/wiki/Black_Swan_(film)>
  - Natalie Portman swans off with best actress Oscar | Oscars 2011 ...: <https://www.theguardian.com/film/2011/feb/28/natalie-portman-oscars-2011-best-actress>
  - Black Swan (2010) - Awards - IMDb: <https://www.imdb.com/title/tt0947798/awards/>
  - Black Swan | Oscars Wiki | Fandom: <https://oscars.fandom.com/wiki/Black_Swan>
- [ ] Question wording and time anchor are clear
- [ ] Expected answer and aliases are factually correct
- [ ] Committed support document proves the answer or intended abstention
- [ ] Typing can plausibly stabilize as classified
- [ ] Stabilization class accepted or corrected without model outputs
- [ ] Split assignment accepted (development or test)

## crag-text-test-005 — test

- **Words:** 21
- **Domain / type / dynamism:** music / simple_w_condition / static
- **Query:** what album did the killers release in 2004, which included the songs "mr. brightside" and "jenny was a friend of mine"?
- **Candidate stabilization class:** early_stabilization
- **Classification confidence:** low
- **Heuristic reason:** a non-generic lexical anchor appears in the first half without a late constraint
- **Gold:** Hot Fuss
- **Aliases:** the album Hot Fuss, The Killers released the album "Hot Fuss" in 2004, which included the songs "Mr. Brightside" and "Jenny Was a Friend of Mine".
- **Own-page evidence covered:** yes
- **Proposed supporting document IDs:** crag-global-980bbdc9e21c1c3d8619078d
- **Original CRAG pages:**
  - The Killers: Hot Fuss Album Review | Pitchfork: <https://pitchfork.com/reviews/albums/4579-hot-fuss/>
  - What is The Killers song 'Mr Brightside' about?: <https://faroutmagazine.co.uk/story-behind-song-the-killers-mr-brightside/>
  - Hot Fuss — The Killers | Last.fm: <https://www.last.fm/music/The+Killers/Hot+Fuss>
  - Hot Fuss - Wikipedia: <https://en.wikipedia.org/wiki/Hot_Fuss>
  - Mr. Brightside by The Killers - Songfacts: <https://www.songfacts.com/facts/the-killers/mr-brightside>
- [ ] Question wording and time anchor are clear
- [ ] Expected answer and aliases are factually correct
- [ ] Committed support document proves the answer or intended abstention
- [ ] Typing can plausibly stabilize as classified
- [ ] Stabilization class accepted or corrected without model outputs
- [ ] Split assignment accepted (development or test)

## crag-text-test-006 — test

- **Words:** 19
- **Domain / type / dynamism:** music / simple_w_condition / static
- **Query:** what album did kings of leon release in 2013, which included the songs "wait for me" and "family tree"?
- **Candidate stabilization class:** early_stabilization
- **Classification confidence:** low
- **Heuristic reason:** a non-generic lexical anchor appears in the first half without a late constraint
- **Gold:** Mechanical Bull
- **Aliases:** the album Mechanical Bull, Kings of Leon released the album "Mechanical Bull" in 2013, which included the songs "Wait for Me" and "Family Tree".
- **Own-page evidence covered:** yes
- **Proposed supporting document IDs:** crag-global-67b771c4dd17ec0e2fc485b6
- **Original CRAG pages:**
  - "FAMILY TREE" LYRICS by KINGS OF LEON: I tell you now...: <https://www.flashlyrics.com/lyrics/kings-of-leon/family-tree-79>
  - Mechanical Bull by Kings of Leon (Album, Alternative Rock): Reviews, ...: <https://rateyourmusic.com/release/album/kings-of-leon/mechanical-bull/>
  - Kings of Leon - Mechanical Bull Album Reviews, Songs & More | AllMusic: <https://www.allmusic.com/album/mw0002569539>
  - Family Tree Lyrics Kings of Leon Song: <https://www.quedeletras.com/family-tree/kings-of-leon/lyrics/167904.html>
  - Kings Of Leon Mechanical Bull Full Album LEAKED [www.mp3zer.com] ...: <https://www.dailymotion.com/video/x12xbnq>
- [ ] Question wording and time anchor are clear
- [ ] Expected answer and aliases are factually correct
- [ ] Committed support document proves the answer or intended abstention
- [ ] Typing can plausibly stabilize as classified
- [ ] Stabilization class accepted or corrected without model outputs
- [ ] Split assignment accepted (development or test)

## crag-text-test-007 — test

- **Words:** 9
- **Domain / type / dynamism:** open / simple_w_condition / slow-changing
- **Query:** what is the most active volcano in the philippines?
- **Candidate stabilization class:** early_stabilization
- **Classification confidence:** low
- **Heuristic reason:** a non-generic lexical anchor appears in the first half without a late constraint
- **Gold:** Mayon Volcano
- **Aliases:** Mayon
- **Own-page evidence covered:** yes
- **Proposed supporting document IDs:** crag-global-c0298641c65cc39b09f74477
- **Original CRAG pages:**
  - Active volcanoes and eruptions in the Philippines: <https://www.worlddata.info/asia/philippines/volcanoes.php>
  - Volcanoes of the Philippines: <https://www.phivolcs.dost.gov.ph/index.php/volcano-hazard/volcanoes-of-the-philippines>
  - What are the Most Active Volcanoes in the Philippines?: <https://www.philippinetraveler.com/active-volcanoes-in-the-philippines/>
  - Volcanoes of Luzon, Philippines: facts & information / Volcano...: <https://www.volcanodiscovery.com/philippines/luzon.html>
- [ ] Question wording and time anchor are clear
- [ ] Expected answer and aliases are factually correct
- [ ] Committed support document proves the answer or intended abstention
- [ ] Typing can plausibly stabilize as classified
- [ ] Stabilization class accepted or corrected without model outputs
- [ ] Split assignment accepted (development or test)

## crag-text-test-008 — test

- **Words:** 10
- **Domain / type / dynamism:** open / set / static
- **Query:** what are the names of george and amal clooney's twins?
- **Original source query:** what are the names of george and amal clooney’s twins?
- **Candidate stabilization class:** late_stabilization
- **Classification confidence:** low
- **Heuristic reason:** the first lexical anchor or a material constraint occurs late in the query
- **Gold:** Ella and Alexander Clooney
- **Aliases:** Ella and Alexander
- **Own-page evidence covered:** yes
- **Proposed supporting document IDs:** crag-global-f3d09c83e3f8c21ca94cbab5
- **Original CRAG pages:**
  - George and Amal Clooney's twins doted upon following 6th birthday ...: <https://www.hellomagazine.com/healthandbeauty/mother-and-baby/498836/george-clooney-amal-clooney-dote-on-twins-during-family-vacation-following-6-birthday-photos/>
  - George Clooney & Amal’s Twins: What to Know About Ella & Alexander ...: <https://www.sheknows.com/feature/george-clooney-amal-twins-ella-alexander-2269671/>
  - George and Amal Clooney's rarely-seen twins' alternative tastes ...: <https://www.hellomagazine.com/healthandbeauty/mother-and-baby/503576/george-and-amal-clooney-rarely-seen-twins-alternative-tastes-revealed/>
  - George and Amal Clooney's Kids, Alexander and Ella | POPSUGAR ...: <https://www.popsugar.com/celebrity/george-amal-clooney-kids-48035171>
  - George and Amal Clooney's Twins Have Totally Different Personalities: <https://www.harpersbazaar.com/celebrity/latest/a34930616/george-clooney-talks-twin-personalities/>
- [ ] Question wording and time anchor are clear
- [ ] Expected answer and aliases are factually correct
- [ ] Committed support document proves the answer or intended abstention
- [ ] Typing can plausibly stabilize as classified
- [ ] Stabilization class accepted or corrected without model outputs
- [ ] Split assignment accepted (development or test)

## crag-text-test-009 — test

- **Words:** 13
- **Domain / type / dynamism:** sports / set / slow-changing
- **Query:** As of March 2024, what NFL teams had never made the Super Bowl?
- **Original source query:** what nfl teams have never made the super bowl?
- **Candidate stabilization class:** late_stabilization
- **Classification confidence:** low
- **Heuristic reason:** the first lexical anchor or a material constraint occurs late in the query
- **Gold:** Cleveland Browns, Detroit Lions, Houston Texans, and Jacksonville Jaguars
- **Aliases:** Browns, Lions, Texans, and Jaguars, Browns, Lions, Jaguars, Texans
- **Own-page evidence covered:** yes
- **Proposed supporting document IDs:** crag-global-6abd5e9a40b58f788f823cc1
- **Original CRAG pages:**
  - Which NFL Teams Have Never Won A Super Bowl? (Updated 2024): <https://www.profootballnetwork.com/nfl-teams-without-a-super-bowl/>
  - NFL teams that have never won a Super Bowl?: <https://sportsnaut.com/the-nfl-teams-that-have-never-won-a-super-bowl/>
  - Which NFL teams have never played in the Super Bowl? - DraftKings ...: <https://dknetwork.draftkings.com/nfl/24053473/super-bowl-history-which-nfl-teams-who-has-never-played-how-many-lions-browns-jaguars-texans>
  - Which NFL teams have never been to the Super Bowl?: <https://www.sportskeeda.com/nfl/which-nfl-teams-never-super-bowl>
- [ ] Question wording and time anchor are clear
- [ ] Expected answer and aliases are factually correct
- [ ] Committed support document proves the answer or intended abstention
- [ ] Typing can plausibly stabilize as classified
- [ ] Stabilization class accepted or corrected without model outputs
- [ ] Split assignment accepted (development or test)

## crag-text-test-010 — test

- **Words:** 18
- **Domain / type / dynamism:** sports / set / static
- **Query:** who are the three players with the most home runs in major league baseball history as of 2022?
- **Candidate stabilization class:** late_stabilization
- **Classification confidence:** low
- **Heuristic reason:** the first lexical anchor or a material constraint occurs late in the query
- **Gold:** Barry Bonds, Hank Aaron, and Babe Ruth
- **Aliases:** Barry Bonds, Hank Aaron, Babe Ruth, As of 2022, Barry Bonds, Hank Aaron, and Babe Ruth are the top three players with the most home runs in Major League Baseball history.
- **Own-page evidence covered:** yes
- **Proposed supporting document IDs:** crag-global-fc4e1286518d7798eb3ef0a5
- **Original CRAG pages:**
  - MLB home run records: Most HRs in a game, season and career in ...: <https://www.sportingnews.com/us/mlb/news/mlb-home-run-records-game-season-career/jxhv3v2qkef04hfnu7hkbwnl>
  - List of Major League Baseball all-time leaders in home runs by ...: <https://en.wikipedia.org/wiki/List_of_Major_League_Baseball_all-time_leaders_in_home_runs_by_pitchers>
  - Career Leaders & Records for Home Runs | Baseball-Reference.com: <https://www.baseball-reference.com/leaders/HR_career.shtml>
- [ ] Question wording and time anchor are clear
- [ ] Expected answer and aliases are factually correct
- [ ] Committed support document proves the answer or intended abstention
- [ ] Typing can plausibly stabilize as classified
- [ ] Stabilization class accepted or corrected without model outputs
- [ ] Split assignment accepted (development or test)

## Global corpus checklist

- [ ] The corpus contains 250 complete CRAG source pages; no page is shortened
- [ ] The 15 preselected support documents and 235 distractors are acceptable
- [ ] Query, answer, split, and gold wrapper fields are absent from retrievable documents
- [ ] Test labels remain scorer-only and are not available to either application path
- [ ] The expected 1,514 chunks (256/32) fit the local-Qdrant runtime target
- [ ] Every item review is complete and all requested corrections are resolved
