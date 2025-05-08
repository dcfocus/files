# 📹 YouTube API Integration for Muxi Baby Feed App

This document demonstrates how our application uses the YouTube Data API to serve personalized, time-sensitive, and age-relevant video content to caregivers through a baby tracking app. It supports our request for increased quota usage.

---

## 🔧 Project Overview

Our app, **Muxi**, helps caregivers discover age-appropriate and timely baby care videos by integrating YouTube video search results into a personalized content feed. The content is:

- **Contextualized by baby’s age**
- **Tailored to the current time of day**
- **Randomized across multiple parenting and development topics**

We use the YouTube API to search for relevant content and display the videos directly to the user in-app.

---

## 📲 YouTube API Usage Details

All YouTube API usage is handled in the [`feed.py`](./app/routers/feed.py) module. We use:

- **Search functionality** to fetch relevant videos
- **Pagination support** using `pageToken`
- **SafeSearch filtering** followed by our own keyword filtering
- **Caching** to avoid redundant requests
- **Per-profile query generation** to ensure usage is purposeful and relevant

---

## 🔍 Example: API Query Flow

In `feed.py`, the following process is followed:

1. **Contextual query generation** based on baby profile and time:

   ```python
   query = generate_contextual_query(profile)


2.	Query sent to YouTube API via SourceManager:
result = await source_manager.fetch_from_source("youtube", query, limit, page_token)
videos = result.get("videos", [])


	3.	Filtering inappropriate content manually:

filtered_videos = _filter_inappropriate_content(videos)

	4.	Fallback to mock data if API fails, ensuring app stability.
💡 Personalization Logic
We build queries using:
	•	👶 Baby age in months
	•	🕐 Time of day (e.g., “睡前活动”, “下午活动”)
	•	🎯 Randomized topic keywords from categories like:
	•	Activities: 手眼协调, 语言发展
	•	Education: 认知启蒙
	•	Care: 疾病预防
	•	Parenting: 亲子阅读

Example generated query:
9个月宝宝 手眼协调 睡前活动

🔄 Refresh Logic

Users can refresh their video feed, which:
	•	Regenerates a new randomized topic query
	•	Clears YouTube cache every 3 refreshes to avoid overuse
	•	Sends a fresh YouTube API request to retrieve new content

_cached_youtube_fetch_wrapper.cache_clear()
query = generate_contextual_query(profile, force_new=True)

