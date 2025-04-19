# Feed API Documentation

## Overview

The Feed API provides personalized video content for baby profiles. It supports initial loading, pagination, and refresh operations to provide a continuous stream of age-appropriate content.

## Endpoints

### 1. Initial Feed Fetch

Retrieves the initial set of videos for a baby profile.

**Endpoint:** `POST /api/feed/`

**Authentication:** Required (JWT Bearer Token)

**Request Body:**

```json
{
  "profile_id": "eb3edf90-42e3-4663-9072-1992cf9d06f6",
  "limit": 5  // Optional, defaults to 5
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "cards": [
      {
        "type": "parent_watch",
        "title": "YouTube推荐",
        "content": "两岁宝宝早教游戏",
        "summary": "这个视频展示了适合两岁宝宝的早教游戏，促进宝宝的认知和运动发展...",
        "thumbnail": "https://i.ytimg.com/vi/videoId1/hqdefault.jpg",
        "media_link": "https://www.youtube.com/watch?v=videoId1",
        "embed_url": "https://www.youtube.com/embed/videoId1",
        "channel": "宝宝早教",
        "duration": "10m 30s",
        "view_count": "345.7K",
        "cta": "观看视频"
      }
    ],
    "query": "两岁宝宝",
    "next_page_token": "CDIQAA"  // Save this for pagination
  }
}
```

### 2. Refresh Feed (Pagination)

Loads the next set of videos using the page token from the previous response.

**Endpoint:** `POST /api/feed/refresh`

**Authentication:** Required (JWT Bearer Token)

**Request Body:**

```json
{
  "profile_id": "eb3edf90-42e3-4663-9072-1992cf9d06f6",
  "limit": 5,  // Optional, defaults to 5
  "page_token": "CDIQAA"  // Required - token from previous response
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "cards": [
      // New set of video cards
    ],
    "query": "两岁宝宝",
    "next_page_token": "CGQQAA"  // New token for next refresh
  }
}
```

### 3. Pull-to-Refresh (Get Fresh Content)

For pull-to-refresh functionality, use the main feed endpoint without a page token to get fresh content.

**Endpoint:** `POST /api/feed/`

**Authentication:** Required (JWT Bearer Token)

**Request Body:**

```json
{
  "profile_id": "eb3edf90-42e3-4663-9072-1992cf9d06f6",
  "limit": 5  // Optional, defaults to 5
}
```

**Response:** Same as Initial Feed Fetch

## Implementation Guide

### Frontend Implementation

#### Initial Load

```javascript
async function fetchInitialFeed() {
  const response = await api.post('/api/feed/', {
    profile_id: profileId,
    limit: 5
  });
  
  if (response.data.success) {
    const feedData = response.data.data;
    displayVideos(feedData.cards);
    
    // Store the token for pagination
    nextPageToken = feedData.next_page_token;
  }
}
```

#### Load More (Pagination)

```javascript
async function loadMoreVideos() {
  if (!nextPageToken) return;
  
  const response = await api.post('/api/feed/refresh', {
    profile_id: profileId,
    limit: 5,
    page_token: nextPageToken
  });
  
  if (response.data.success) {
    const feedData = response.data.data;
    appendVideos(feedData.cards);
    
    // Update the token for next pagination
    nextPageToken = feedData.next_page_token;
  }
}
```

#### Pull-to-Refresh

```javascript
async function pullToRefresh() {
  // Reset pagination
  nextPageToken = null;
  
  // Get fresh content
  const response = await api.post('/api/feed/', {
    profile_id: profileId,
    limit: 5
  });
  
  if (response.data.success) {
    const feedData = response.data.data;
    replaceVideos(feedData.cards);
    
    // Store the new token
    nextPageToken = feedData.next_page_token;
  }
}
```

### Flutter Implementation

```dart
// Initial load
Future<void> fetchInitialFeed() async {
  final response = await dio.post('/api/feed/', data: {
    'profile_id': profileId,
    'limit': 5
  });
  
  if (response.data['success']) {
    final feedData = response.data['data'];
    setState(() {
      videos = feedData['cards'];
      nextPageToken = feedData['next_page_token'];
    });
  }
}

// Load more
Future<void> loadMoreVideos() async {
  if (nextPageToken == null) return;
  
  final response = await dio.post('/api/feed/refresh', data: {
    'profile_id': profileId,
    'limit': 5,
    'page_token': nextPageToken
  });
  
  if (response.data['success']) {
    final feedData = response.data['data'];
    setState(() {
      videos.addAll(feedData['cards']);
      nextPageToken = feedData['next_page_token'];
    });
  }
}

// Pull to refresh
Future<void> onRefresh() async {
  nextPageToken = null;
  
  final response = await dio.post('/api/feed/', data: {
    'profile_id': profileId,
    'limit': 5
  });
  
  if (response.data['success']) {
    final feedData = response.data['data'];
    setState(() {
      videos = feedData['cards'];
      nextPageToken = feedData['next_page_token'];
    });
  }
}

// UI implementation
RefreshIndicator(
  onRefresh: onRefresh,
  child: ListView.builder(
    itemCount: videos.length + 1,
    itemBuilder: (context, index) {
      if (index == videos.length) {
        // Load more when reaching the end
        loadMoreVideos();
        return Center(child: CircularProgressIndicator());
      }
      return VideoCard(video: videos[index]);
    },
  ),
)
```

## Error Handling

All endpoints return a consistent error format:

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid profile ID"
  }
}
```

Common error codes:
- `AUTHENTICATION_ERROR`: Authentication issues
- `VALIDATION_ERROR`: Invalid request data
- `NOT_FOUND`: Resource not found
- `PERMISSION_DENIED`: User doesn't have permission
- `SERVER_ERROR`: Internal server error
