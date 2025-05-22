from fastapi import APIRouter, Depends, HTTPException, Query, Body
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from functools import lru_cache
import time
import logging
from pydantic import BaseModel
import traceback
import json
import random
import pytz
import os
import numpy as np
from google import genai
from google.genai.types import EmbedContentConfig
import google.generativeai as genai_flash

from app.database import get_db
from app.dependencies import get_current_user
from app.models.models import Profile, Feeding, Diaper, Sleep, Bath, Growth, Settings
#from app.agents.feed_orchestrator import FeedOrchestratorAgent
from app.feeds_sources.manager import SourceManager
from app.feeds_sources.youtube import QuotaExceededError

print(genai.__version__)

# Set up logger
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/feed",
    tags=["feed"]
)

# Initialize the source manager
source_manager = SourceManager()

# Cache YouTube results for 1 hour
@lru_cache(maxsize=32)
def _cached_youtube_fetch_wrapper(query: str, limit: int, timestamp: int):
    """Wrapper function that returns a fresh coroutine each time"""
    # This function returns a new coroutine each time it's called
    return source_manager.fetch_from_source("youtube", query, limit)

async def get_cached_youtube_videos(query: str, limit: int):
    """Get YouTube videos with caching"""
    # Create hourly timestamp for cache invalidation
    timestamp = int(time.time() / 3600)
    try:
        # Get a fresh coroutine from the cache wrapper
        coroutine = _cached_youtube_fetch_wrapper(query, limit, timestamp)
        # Await the coroutine to get the actual results
        return await coroutine
    except Exception as e:
        logger.error(f"Error fetching YouTube videos: {str(e)}")
        # Return empty list on error
        return []

def get_text_embeddings(texts):
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.embed_content(
        model="models/embedding-001",
        contents=texts
    )
    return [np.array(embedding.values) for embedding in response.embeddings]

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def dot_product_similarity(a, b):
    """Calculate dot product similarity between two vectors."""
    return np.dot(a, b)

# Define request model
class FeedRequest(BaseModel):
    """
    Request model for feed endpoints
    """
    profile_id: str
    limit: Optional[int] = 5  # Default to 5 videos
    page_token: Optional[str] = None  # For pagination with YouTube API

async def get_youtube_videos(query: str, limit: int):
    """Get YouTube videos without caching"""
    try:
        # Directly fetch from source manager without caching
        result = await source_manager.fetch_from_source("youtube", query, limit)
        
        # Ensure result is a list of dictionaries
        if not isinstance(result, list):
            logger.error(f"Unexpected result type from YouTube source: {type(result)}")
            return []
            
        # Check each item in the list
        videos = []
        for item in result:
            if isinstance(item, dict):
                videos.append(item)
            else:
                logger.warning(f"Skipping non-dictionary item in YouTube results: {type(item)}")
                
        return videos
    except Exception as e:
        logger.error(f"Error fetching YouTube videos: {str(e)}")
        # Return empty list on error
        return []

# Define baby-related topics
BABY_TOPICS = {
    "activities": [
        "早教游戏", "亲子互动", "音乐启蒙", "运动发展", "认知训练",
        "语言发展", "感官刺激", "手眼协调", "大运动", "精细动作"
    ],
    "care": [
        "日常护理", "营养饮食", "睡眠训练", "情绪管理", "安全防护",
        "疾病预防", "生长发育", "疫苗接种", "牙齿护理", "皮肤护理"
    ],
    "education": [
        "认知启蒙", "语言学习", "数学启蒙", "艺术启蒙", "科学探索",
        "社交能力", "自理能力", "专注力训练", "创造力培养", "情商培养"
    ],
    "parenting": [
        "育儿经验", "亲子关系", "行为引导", "情绪管理", "家庭互动",
        "教育方法", "成长记录", "亲子阅读", "户外活动", "家庭游戏"
    ]
}

# Define time periods
TIME_PERIODS = {
    "early_morning": (5, 8),    # 5:00 - 8:00
    "morning": (8, 11),         # 8:00 - 11:00
    "noon": (11, 14),           # 11:00 - 14:00
    "afternoon": (14, 17),      # 14:00 - 17:00
    "evening": (17, 20),        # 17:00 - 20:00
    "night": (20, 23),          # 20:00 - 23:00
    "late_night": (23, 5)       # 23:00 - 5:00
}

# Add this after other constants
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

def get_current_time_period():
    """Get the current time period based on the hour in Los Angeles timezone."""
    # Get Los Angeles timezone
    la_tz = pytz.timezone('America/Los_Angeles')
    # Get current time in LA
    current_time = datetime.now(la_tz)
    current_hour = current_time.hour
    
    for period, (start, end) in TIME_PERIODS.items():
        if start <= end:
            if start <= current_hour < end:
                return period
        else:  # Handle overnight period
            if current_hour >= start or current_hour < end:
                return period
    
    return "morning"  # Default fallback

def get_age_based_query(age_months):
    """Get the base query based on age."""
    if age_months < 3:  # 0-3 months
        return "新生儿"
    elif age_months < 6:  # 3-6 months
        return "3个月宝宝"
    elif age_months < 9:  # 6-9 months
        return "6个月宝宝"
    elif age_months < 12:  # 9-12 months
        return "9个月宝宝"
    elif age_months < 15:  # 12-15 months
        return "一岁宝宝"
    elif age_months < 18:  # 15-18 months
        return "15个月宝宝"
    elif age_months < 21:  # 18-21 months
        return "18个月宝宝"
    elif age_months < 24:  # 21-24 months
        return "21个月宝宝"
    elif age_months < 30:  # 2-2.5 years
        return "两岁宝宝"
    elif age_months < 36:  # 2.5-3 years
        return "两岁半宝宝"
    elif age_months < 42:  # 3-3.5 years
        return "三岁宝宝"
    elif age_months < 48:  # 3.5-4 years
        return "三岁半宝宝"
    elif age_months < 54:  # 4-4.5 years
        return "四岁宝宝"
    elif age_months < 60:  # 4.5-5 years
        return "四岁半宝宝"
    else:  # 5+ years
        return "五岁宝宝"

def get_time_based_context(time_period):
    """Get time-appropriate context for the query."""
    time_contexts = {
        "early_morning": "晨间活动",
        "morning": "上午活动",
        "noon": "午间活动",
        "afternoon": "下午活动",
        "evening": "晚间活动",
        "night": "睡前活动",
        "late_night": "夜间护理"
    }
    return time_contexts.get(time_period, "")

def generate_query(profile, language='en', force_new=True):
    """
    Generate a contextual query combining age, topic, and time.
    
    Args:
        profile: The baby profile
        language: Language preference ('en' or 'zh')
        force_new: If True, ensures a new random topic is generated
    """
    # Calculate age in months
    age_days = (datetime.now() - profile.birth_date).days if profile.birth_date else 365
    age_months = age_days // 30  # Approximate conversion
    
    # Get base age query based on language
    if language == 'en':
        age_query = get_age_based_query_en(age_months)
    else:
        age_query = get_age_based_query(age_months)
    
    # Get current time period (deterministic based on current time)
    time_period = get_current_time_period()
    
    # Get time context based on language
    if language == 'en':
        time_context = get_time_based_context_en(time_period)
    else:
        time_context = get_time_based_context(time_period)
    
    # Randomly select a topic category and specific topic
    if force_new:
        random.seed(datetime.now().timestamp())
    
    # Select topic based on language
    if language == 'en':
        topic_category = random.choice(list(BABY_TOPICS_EN.keys()))
        topic = random.choice(BABY_TOPICS_EN[topic_category])
    else:
        topic_category = random.choice(list(BABY_TOPICS.keys()))
        topic = random.choice(BABY_TOPICS[topic_category])
    
    # Combine all elements into a query
    query_parts = [age_query, topic, time_context]
    query = " ".join(filter(None, query_parts))
    
    # Log the query generation for debugging
    logger.info(f"Generated contextual query ({language}): {query}")
    logger.info(f"Query components - Age: {age_query}, Topic: {topic}, Time: {time_context}")
    logger.info(f"Time period: {time_period} (deterministic based on current time)")
    
    return query

@router.post("/")
async def get_feed(
    request: FeedRequest,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Generate a personalized feed for the specified baby profile
    """
    try:
        # Extract profile_id from request body
        profile_id = request.profile_id
        
        # Log for debugging
        logger.info(f"Processing feed request for profile_id: {profile_id}")
        
        # 1. Verify profile belongs to current user
        profile = db.query(Profile).filter(
            Profile.server_id == profile_id,
            Profile.account_id == current_user.id
        ).first()
        
        if not profile:
            logger.warning(f"Profile not found: {profile_id} for user {current_user.id}")
            raise HTTPException(
                status_code=404,
                detail="Profile not found"
            )
        
        # Get user's language preference
        settings = db.query(Settings).filter(Settings.profile_id == profile_id).first()
        language = settings.language if settings else 'en'  # Default to English if no settings
        
        # 2. Gather context for the feed
        logger.info(f"Building context for profile: {profile.name}")
        context = await _build_baby_context(profile, db)
        logger.info("Context built successfully")
        
        # 3. Generate contextual query using age, topic, and time
        query = generate_query(profile, language=language)
        logger.info(f"Generated contextual query: {query}")
        
        # 4. Try to fetch YouTube videos, fall back to mock data if it fails
        try:
            # Get videos with the specified limit
            result = await source_manager.fetch_from_source("youtube", query, request.limit, request.page_token)
            videos = result.get("videos", [])
            next_page_token = result.get("next_page_token")  # Get the token from result
            
            logger.info(f"Retrieved {len(videos)} videos from YouTube API")
            
            # Filter out inappropriate content
            videos = _filter_inappropriate_content(videos)
            logger.info(f"After content filtering: {len(videos)} videos remaining")
            
            # Validate videos format
            if not videos or not isinstance(videos, list):
                logger.warning("Invalid response from YouTube API, using mock data")
                raise ValueError("Invalid YouTube API response")
                
            # Check first video to ensure it's a dictionary
            if videos and not isinstance(videos[0], dict):
                logger.warning(f"YouTube API returned non-dictionary items: {type(videos[0])}")
                raise ValueError("Invalid YouTube API response format")
                
        except Exception as e:
            logger.warning(f"Error fetching from YouTube API: {str(e)}, using mock data")
            # Use mock data as fallback
            videos = [
                {
                    "id": "video1",
                    "title": "如何照顾两岁宝宝的日常护理指南",
                    "description": "这是一个关于两岁宝宝的视频，提供了实用的育儿技巧和建议。",
                    "thumbnail": "https://img.youtube.com/vi/video1/hqdefault.jpg",
                    "channel": "宝宝成长记",
                    "published_at": "2023-05-15T14:30:00Z",
                    "url": "https://www.youtube.com/watch?v=video1",
                    "embed_url": "https://www.youtube.com/embed/video1",
                    "duration": "10m 30s",
                    "view_count": "345,678",
                    "source": "youtube"
                },
                {
                    "id": "video2",
                    "title": "两岁宝宝早教游戏：促进认知发展",
                    "description": "这个视频展示了适合两岁宝宝的早教游戏，帮助促进认知发展和语言能力。",
                    "thumbnail": "https://img.youtube.com/vi/video2/hqdefault.jpg",
                    "channel": "育儿专家谈",
                    "published_at": "2023-06-20T10:15:00Z",
                    "url": "https://www.youtube.com/watch?v=video2",
                    "embed_url": "https://www.youtube.com/embed/video2",
                    "duration": "8m 45s",
                    "view_count": "245,678",
                    "source": "youtube"
                },
                {
                    "id": "video3",
                    "title": "两岁宝宝饮食营养搭配方案",
                    "description": "专业营养师分享两岁宝宝的饮食营养搭配方案，确保宝宝健康成长。",
                    "thumbnail": "https://img.youtube.com/vi/video3/hqdefault.jpg",
                    "channel": "健康宝宝",
                    "published_at": "2023-07-05T09:45:00Z",
                    "url": "https://www.youtube.com/watch?v=video3",
                    "embed_url": "https://www.youtube.com/embed/video3",
                    "duration": "12m 20s",
                    "view_count": "198,432",
                    "source": "youtube"
                },
                {
                    "id": "video4",
                    "title": "两岁宝宝常见疾病预防与护理",
                    "description": "儿科医生讲解两岁宝宝常见疾病的预防措施和护理方法，帮助家长更好地照顾宝宝。",
                    "thumbnail": "https://img.youtube.com/vi/video4/hqdefault.jpg",
                    "channel": "儿科医生在线",
                    "published_at": "2023-08-10T16:20:00Z",
                    "url": "https://www.youtube.com/watch?v=video4",
                    "embed_url": "https://www.youtube.com/embed/video4",
                    "duration": "15m 10s",
                    "view_count": "321,654",
                    "source": "youtube"
                },
                {
                    "id": "video5",
                    "title": "两岁宝宝语言发展训练方法",
                    "description": "语言治疗师分享促进两岁宝宝语言发展的有效训练方法和互动游戏。",
                    "thumbnail": "https://img.youtube.com/vi/video5/hqdefault.jpg",
                    "channel": "宝宝语言发展",
                    "published_at": "2023-09-15T11:30:00Z",
                    "url": "https://www.youtube.com/watch?v=video5",
                    "embed_url": "https://www.youtube.com/embed/video5",
                    "duration": "9m 55s",
                    "view_count": "276,543",
                    "source": "youtube"
                }
            ]
        
        # 5. Format videos as feed cards
        video_cards = format_videos_as_cards(videos, profile)
        
        # Get current time period and age query for summary
        time_period = get_current_time_period()
        age_days = (datetime.now() - profile.birth_date).days if profile.birth_date else 365
        age_months = age_days // 30
        age_query = get_age_based_query(age_months)

        # Generate feed summary using Gemini
        feed_summary = await generate_feed_summary(query, time_period, age_query, context, language)
        
        return {
            "success": True,
            "data": {
                "feed_summary": feed_summary,
                "cards": video_cards,
                "query": query,
                "next_page_token": next_page_token
            }
        }
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log the full traceback for debugging
        logger.error(f"Error in get_feed: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error generating feed: {str(e)}"
        )

@router.post("/youtube")
async def get_youtube_videos(
    query: str = Query(None, description="Search query"),
    limit: int = Query(5, description="Number of videos to return"),
    page_token: Optional[str] = Query(None, description="Page token for pagination"),
    profile_id: Optional[str] = Query(None, description="Profile ID for language preference"),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Fetch YouTube videos based on the query
    
    This endpoint is a direct pass-through to the YouTube API,
    allowing the app to get videos without the full feed generation.
    """
    try:
        # If no query provided, generate one based on profile
        if not query and profile_id:
            profile = db.query(Profile).filter(
                Profile.server_id == profile_id,
                Profile.account_id == current_user.id
            ).first()
            
            if profile:
                # Get user's language preference
                settings = db.query(Settings).filter(Settings.profile_id == profile_id).first()
                language = settings.language if settings else 'en'  # Default to English if no settings
                query = generate_query(profile, language=language)
            else:
                query = "baby care"  # Default fallback query
        
        # Initial request gets first page and nextPageToken
        result = await source_manager.fetch_from_source("youtube", query, limit)
        videos = result.get("videos", [])
        next_page_token = result.get("next_page_token")
        
        # Filter out inappropriate content
        filtered_videos = _filter_inappropriate_content(videos)
        
        # Use Gemini embedding to re-rank by semantic similarity to the query
        try:
            if filtered_videos:
                # 1. Get video titles
                video_titles = [v.get("title", "") for v in filtered_videos]
                logger.info(f"Processing {len(video_titles)} videos for semantic ranking")

                # 2. Get query embedding
                query_embedding = get_text_embeddings([query])[0]
                logger.info(f"Generated embedding for query: {query}")

                # 3. Get title embeddings
                title_embeddings = get_text_embeddings(video_titles)
                logger.info(f"Generated embeddings for {len(title_embeddings)} video titles")

                # 4. Compute similarity and rank
                cosine_scores = [cosine_similarity(query_embedding, emb) for emb in title_embeddings]
                dot_scores = [dot_product_similarity(query_embedding, emb) for emb in title_embeddings]
                
                # Log both scores for comparison
                logger.info("Video ranking comparison (Cosine vs Dot Product):")
                for i, (video, cos_score, dot_score) in enumerate(zip(filtered_videos, cosine_scores, dot_scores)):
                    logger.info(f"Rank {i+1}: Cosine={cos_score:.4f}, Dot={dot_score:.4f}, Title='{video.get('title', 'No title')}'")
                
                # Use dot product for final ranking
                ranked_videos = sorted(zip(filtered_videos, dot_scores), key=lambda x: x[1], reverse=True)
                
                # Log detailed ranking information
                logger.info("Video ranking by semantic similarity:")
                for i, (video, score) in enumerate(ranked_videos):
                    logger.info(f"Rank {i+1}: Score={score:.4f}, Title='{video.get('title', 'No title')}'")
                
                # 5. Keep top 5 only
                filtered_videos = [video for video, _ in ranked_videos[:5]]
                logger.info(f"Selected top 5 videos after semantic ranking")
        except Exception as e:
            logger.error(f"Error ranking videos by Gemini embedding similarity: {str(e)}")
            logger.error(traceback.format_exc())
        
        # Return next_page_token to client
        response_data = {
            "success": True,
            "data": {
                "videos": filtered_videos,
                "query": query,
                "next_page_token": next_page_token
            }
        }
        
        return response_data
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching YouTube videos: {str(e)}"
        )

@router.post("/refresh/")
async def refresh_feed(
    request: FeedRequest,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Refresh the feed with new content for the specified baby profile
    """
    try:
        # Extract profile_id, limit, and page_token from request body
        profile_id = request.profile_id
        limit = request.limit or 5
        page_token = request.page_token

        # Log request details
        logger.info(f"Refresh request - profile_id: {profile_id}, limit: {limit}, page_token: {page_token}")

        # 1. Verify profile belongs to current user
        profile = db.query(Profile).filter(
            Profile.server_id == profile_id,
            Profile.account_id == current_user.id
        ).first()

        if not profile:
            raise HTTPException(
                status_code=404,
                detail="Profile not found"
            )

        # Get user's language preference
        settings = db.query(Settings).filter(Settings.profile_id == profile_id).first()
        language = settings.language if settings else 'zh'  # Default to Chinese if no settings

        # 2. Generate contextual query using age, topic, and time
        query = generate_query(profile, language=language)
        logger.info(f"Generated contextual query for refresh: {query}")

        context = await _build_baby_context(profile, db)
        logger.info("Context built successfully")

        # Ensure cache is cleared and a new random query is generated if needed
        _cached_youtube_fetch_wrapper.cache_clear()
        query = generate_query(profile, language=language, force_new=True)
        logger.info(f"Regenerated contextual query after cache clear: {query}")
        print(f"Regenerated contextual query after cache clear: {query}")

        try:
            if page_token:
                # Use the provided page_token to get the next page of results
                result = await source_manager.fetch_from_source("youtube", query, limit, page_token)
                videos = result.get("videos", [])
                next_page_token = result.get("next_page_token")

                logger.info(f"YouTube API response with page_token - videos count: {len(videos)}")
                logger.info(f"Next page token: {next_page_token}")
            else:
                # If no page_token, use the contextual query directly
                result = await source_manager.fetch_from_source("youtube", query, limit)
                videos = result.get("videos", [])
                next_page_token = result.get("next_page_token")

                logger.info(f"YouTube API response without page_token - videos count: {len(videos)}")
                logger.info(f"Next page token: {next_page_token}")

            # Log first video for debugging
            if videos:
                logger.info(f"First video title: {videos[0].get('title', 'No title')}")
                logger.info(f"First video ID: {videos[0].get('id', 'No ID')}")

        except Exception as e:
            logger.error(f"Error fetching from YouTube API: {str(e)}")
            logger.error(traceback.format_exc())
            # ... fallback to mock data ...

        # Filter out inappropriate content
        filtered_videos = _filter_inappropriate_content(videos)

        # Use Gemini embedding to re-rank by semantic similarity to the query
        try:
            if filtered_videos:
                # 1. Get video titles
                video_titles = [v.get("title", "") for v in filtered_videos]
                logger.info(f"Processing {len(video_titles)} videos for semantic ranking")

                # 2. Get query embedding
                query_embedding = get_text_embeddings([query])[0]
                logger.info(f"Generated embedding for query: {query}")

                # 3. Get title embeddings
                title_embeddings = get_text_embeddings(video_titles)
                logger.info(f"Generated embeddings for {len(title_embeddings)} video titles")

                # 4. Compute similarity and rank
                cosine_scores = [cosine_similarity(query_embedding, emb) for emb in title_embeddings]
                dot_scores = [dot_product_similarity(query_embedding, emb) for emb in title_embeddings]
                
                # Log both scores for comparison
                logger.info("Video ranking comparison (Cosine vs Dot Product):")
                for i, (video, cos_score, dot_score) in enumerate(zip(filtered_videos, cosine_scores, dot_scores)):
                    logger.info(f"Rank {i+1}: Cosine={cos_score:.4f}, Dot={dot_score:.4f}, Title='{video.get('title', 'No title')}'")
                
                # Use dot product for final ranking
                ranked_videos = sorted(zip(filtered_videos, dot_scores), key=lambda x: x[1], reverse=True)
                
                # Log detailed ranking information
                logger.info("Video ranking by semantic similarity:")
                for i, (video, score) in enumerate(ranked_videos):
                    logger.info(f"Rank {i+1}: Score={score:.4f}, Title='{video.get('title', 'No title')}'")
                
                # 5. Keep top 5 only
                #filtered_videos = [video for video, _ in ranked_videos[:5]]
                filtered_videos = [video for video, _ in ranked_videos]
                logger.info(f"Selected top 5 videos after semantic ranking")
        except Exception as e:
            logger.error(f"Error ranking videos by Gemini embedding similarity: {str(e)}")
            logger.error(traceback.format_exc())

        # 3. Format videos as feed cards
        video_cards = format_videos_as_cards(filtered_videos, profile)

        # Get current time period and age query for summary
        time_period = get_current_time_period()
        age_days = (datetime.now() - profile.birth_date).days if profile.birth_date else 365
        age_months = age_days // 30
        age_query = get_age_based_query_en(age_months) if language == 'en' else get_age_based_query(age_months)

        # Generate feed summary using Gemini
        feed_summary = await generate_feed_summary(query, time_period, age_query, context, language)
        
        response_data = {
            "success": True,
            "data": {
                "feed_summary": feed_summary,
                "cards": video_cards,
                "query": query,
                "next_page_token": next_page_token
            }
        }

        # Include next_page_token in response if available
        if next_page_token:
            response_data["data"]["next_page_token"] = next_page_token
            logger.info(f"Returning response with next_page_token: {next_page_token}")
        else:
            logger.info("No next_page_token in response")

        # Log final response structure (without full video content)
        logger.info(f"Response structure: {json.dumps({
            'success': response_data['success'],
            'data': {
                'cards_count': len(response_data['data']['cards']),
                'query': response_data['data']['query'],
                'has_next_page': 'next_page_token' in response_data['data']
            }
        }, indent=2)}")

        return response_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in refresh_feed: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error refreshing feed: {str(e)}"
        )

async def generate_feed_summary(query: str, time_period: str, age_query: str, context: Dict[str, Any], language: str = 'zh') -> str:
    """
    Generate a natural language summary of the feed using Google Gemini 2.0 Flash.
    
    Args:
        query: The search query used
        time_period: Current time period (e.g., "morning", "afternoon")
        age_query: Age-based query (e.g., "一岁宝宝")
        context: Dictionary containing baby's context information
        language: Language preference ('en' or 'zh')
        
    Returns:
        A natural language summary of the feed
    """
    try:
        if not GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY not found in environment variables")
            return f"Showing videos filtered by keyword '{query}'"
        
        # Configure Gemini Flash
        genai_flash.configure(api_key=GEMINI_API_KEY)
        model = genai_flash.GenerativeModel("gemini-1.5-flash")
        
        # Format activities summary
        activities_text = "\n".join(context["activities_summary"]) if context["activities_summary"] else "暂无记录"
        
        # Format measurements text
        measurements = []
        if context.get('height'):
            measurements.append(f"身高 {context['height']}cm")
        if context.get('weight'):
            measurements.append(f"体重 {context['weight']}kg")
        measurements_text = "，".join(measurements) if measurements else "暂无记录"
        
        # Format growth information
        growth_info = []
        if context.get('latest_growth'):
            growth = context['latest_growth']
            if growth.get('height'):
                growth_info.append(f"最新身高 {growth['height']}cm")
            if growth.get('weight'):
                growth_info.append(f"最新体重 {growth['weight']}kg")
            if growth.get('head_circumference'):
                growth_info.append(f"最新头围 {growth['head_circumference']}cm")
            if growth.get('notes'):
                growth_info.append(f"备注：{growth['notes']}")
        growth_text = "，".join(growth_info) if growth_info else "暂无记录"
        
        # Format gender text
        gender_text = {
            'boy': '男宝宝' if language == 'zh' else 'baby boy',
            'girl': '女宝宝' if language == 'zh' else 'baby girl',
            'other': '宝宝' if language == 'zh' else 'baby',
            'unknown': '宝宝' if language == 'zh' else 'baby'
        }.get(context.get('gender', 'unknown'), '宝宝' if language == 'zh' else 'baby')
        
        # Create prompt for Gemini
        if language == 'en':
            prompt = f"""
            You are a professional parenting assistant. Please analyze the baby's recent data and provide professional insights.
            Then generate 1-2 sentences to attract caregivers to watch the recommended videos:

            Content Information:
            - Baby's name: {context['name']}
            - Baby's gender: {gender_text}
            - Baby's age: {age_query} ({context['age_months']} months, {context['age_days']} days)
            - Baby's measurements: {measurements_text}
            - Latest growth data: {growth_text}
            - Current time: {time_period}
            - Video keywords: {query}
            - Baby's recent activities:
            {activities_text}

            Note:
            1. If keywords conflict with baby's age or time period, use baby's age and time period as ground truth
            2. If there's no recent baby data, generate reasonable conclusions based on age and time period
            3. Pay attention to key milestones like lactation consultant in the first two months
            4. Note important growth milestones like introducing solid foods
            5. Remind about pediatrician visits, daycare enrollment, and vaccinations
            6. Based on growth data, provide relevant suggestions and focus points. If no new growth data, remind parents to record growth data

            Generation steps:
            1. First greet parents warmly, using the baby's name
            2. Based on recent data and growth, provide professional analysis and suggestions
            3. Generate 1-2 engaging sentences based on baby's current age and state
            4. Keep the tone warm, professional, and friendly
            5. Keep content concise and focused

            Requirements:
            1. Use warm and friendly tone, feel free to use emojis
            2. Emphasize value to parents, highlight importance of video content for baby's growth
            3. 2-5 sentences in length, ensure information is complete and easy to understand
            4. Use natural English expressions
            5. Mention relevant growth milestones based on baby's age
            6. Be creative while maintaining professionalism and accuracy
            """
        else:
            prompt = f"""
            你是一个专业的育儿助手。请结合宝宝近期的数据得出专业的结论。 
            然后基于以下内容生成1-2句话的邀请, 从而吸引看护着观看下面推送的视频:

            内容信息：
            - 宝宝名字：{context['name']}
            - 宝宝性别：{gender_text}
            - 宝宝年龄：{age_query}（{context['age_months']}个月，{context['age_days']}天）
            - 宝宝数据：{measurements_text}
            - 最新生长数据：{growth_text}
            - 现在的时间是：{time_period}
            - 推送的视频关键词：{query}
            - 宝宝近期的数据：
            {activities_text}

            注意: 
            1. 如果关键词跟宝宝年龄或者时间段冲突请使用宝宝年龄和时间段作为ground truth
            2. 如果宝宝近期的数据没有，请根据宝宝的年龄和时间段生成一个合理的结论
            3. 注意一些关键的时间点, 比如前两个月可以关注秘书师等
            4. 注意宝宝成长的关键时间点，比如添加辅食等
            5. 以及按时提示找儿医, 排队day care, 疫苗接种等
            6. 根据宝宝的生长数据，给出相应的建议和关注点，如果宝宝没有新的生长数据，可以提示宝宝妈妈可以记录宝宝的生长数据

            生成步骤：
            1. 首先用轻松愉快的语气和宝爸宝妈打招呼，可以称呼宝宝的名字
            2. 根据宝宝近期的数据和生长情况，给出专业的分析和建议
            3. 基于宝宝当前的年龄和状态，生成1-2句吸引人的邀请语，引导看护者观看推荐的视频
            4. 注意语气要温暖、专业、有亲和力
            5. 内容要简洁，重点突出

            要求：
            1. 使用温暖友好的语气，可以适当使用emoji增加亲和力
            2. 突出对父母的价值，强调视频内容对宝宝成长的重要性
            3. 2-5句话的长度，确保信息完整且易于理解
            4. 使用自然的中文表达，避免生硬的翻译腔
            5. 根据宝宝年龄特点，适当提及相关的成长里程碑, 比如添加辅食、泌乳师、儿医、day care等
            6. 可以发挥你的创造力，但保持专业性和准确性
            """
        
        # Generate response
        response = model.generate_content(prompt)
        
        if not response or not response.text:
            logger.warning("Empty response from Gemini API")
            return f"Showing videos filtered by keyword '{query}'"
            
        return response.text
        
    except Exception as e:
        logger.error(f"Error generating feed summary with Gemini: {str(e)}")
        logger.error(traceback.format_exc())
        # Fallback to basic summary
        return f"Showing videos filtered by keyword '{query}'"

# Add English versions of the constants
BABY_TOPICS_EN = {
    "activities": [
        "early education games", "parent-child interaction", "music enlightenment", "motor development", "cognitive training",
        "language development", "sensory stimulation", "hand-eye coordination", "gross motor", "fine motor"
    ],
    "care": [
        "daily care", "nutrition", "sleep training", "emotion management", "safety protection",
        "disease prevention", "growth development", "vaccination", "dental care", "skin care"
    ],
    "education": [
        "cognitive enlightenment", "language learning", "math enlightenment", "art enlightenment", "science exploration",
        "social skills", "self-care ability", "focus training", "creativity cultivation", "emotional intelligence"
    ],
    "parenting": [
        "parenting experience", "parent-child relationship", "behavior guidance", "emotion management", "family interaction",
        "education methods", "growth records", "parent-child reading", "outdoor activities", "family games"
    ]
}

TIME_PERIODS_EN = {
    "early_morning": "early morning routine",
    "morning": "morning activities",
    "noon": "noon activities",
    "afternoon": "afternoon activities",
    "evening": "evening routine",
    "night": "bedtime routine",
    "late_night": "night care"
}

def get_age_based_query_en(age_months):
    """Get the base query based on age in English."""
    if age_months < 3:  # 0-3 months
        return "newborn baby"
    elif age_months < 6:  # 3-6 months
        return "3 month old baby"
    elif age_months < 9:  # 6-9 months
        return "6 month old baby"
    elif age_months < 12:  # 9-12 months
        return "9 month old baby"
    elif age_months < 15:  # 12-15 months
        return "1 year old baby"
    elif age_months < 18:  # 15-18 months
        return "15 month old baby"
    elif age_months < 21:  # 18-21 months
        return "18 month old baby"
    elif age_months < 24:  # 21-24 months
        return "21 month old baby"
    elif age_months < 30:  # 2-2.5 years
        return "2 year old baby"
    elif age_months < 36:  # 2.5-3 years
        return "2.5 year old baby"
    elif age_months < 42:  # 3-3.5 years
        return "3 year old baby"
    elif age_months < 48:  # 3.5-4 years
        return "3.5 year old baby"
    elif age_months < 54:  # 4-4.5 years
        return "4 year old baby"
    elif age_months < 60:  # 4.5-5 years
        return "4.5 year old baby"
    else:  # 5+ years
        return "5 year old baby"

def get_time_based_context_en(time_period):
    """Get time-appropriate context for the query in English."""
    return TIME_PERIODS_EN.get(time_period, "")

async def _build_baby_context(profile: Profile, db: Session) -> Dict[str, Any]:
    """
    Build a comprehensive context about the baby for feed generation
    
    Returns:
        Dict[str, Any]: A dictionary containing baby's context information
    """
    # Initialize age variables outside try block
    age_days = 0
    age_months = 0
    
    try:
        logger.info(f"Building context for profile: {profile.name} (ID: {profile.server_id})")
        
        # Get current time in UTC
        now = datetime.now(pytz.UTC)
        three_days_ago = now - timedelta(days=3)
        logger.info(f"Fetching activities from {three_days_ago} to {now}")
        
        # Check for recent growth records (within 2 days)
        two_days_ago = now - timedelta(days=2)
        recent_growth = db.query(Growth).filter(
            Growth.profile_id == profile.server_id,
            Growth.created_at >= two_days_ago
        ).first()
        
        if recent_growth:
            logger.info(f"Found recent growth record from {recent_growth.created_at}")
        
        # Calculate age in months (handle timezone properly)
        if profile.birth_date:
            # Convert birth_date to UTC if it's naive
            if profile.birth_date.tzinfo is None:
                birth_date = pytz.UTC.localize(profile.birth_date)
            else:
                birth_date = profile.birth_date
            age_days = (now - birth_date).days
            age_months = age_days // 30
            logger.info(f"Baby age: {age_months} months ({age_days} days)")
        else:
            logger.warning("No birth date provided for profile")
            age_days = 180  # Default to 6 months
            age_months = 6
        
        # Get recent activities (filter by created_at timestamp)
        recent_feedings = db.query(Feeding).filter(
            Feeding.profile_id == profile.server_id,
            Feeding.created_at >= three_days_ago
        ).order_by(Feeding.created_at.desc()).all()
        logger.info(f"Found {len(recent_feedings)} recent feedings")
        
        recent_diapers = db.query(Diaper).filter(
            Diaper.profile_id == profile.server_id,
            Diaper.created_at >= three_days_ago
        ).order_by(Diaper.created_at.desc()).all()
        logger.info(f"Found {len(recent_diapers)} recent diapers")
        
        recent_sleeps = db.query(Sleep).filter(
            Sleep.profile_id == profile.server_id,
            Sleep.created_at >= three_days_ago
        ).order_by(Sleep.created_at.desc()).all()
        logger.info(f"Found {len(recent_sleeps)} recent sleep records")
        
        recent_baths = db.query(Bath).filter(
            Bath.profile_id == profile.server_id,
            Bath.created_at >= three_days_ago
        ).order_by(Bath.created_at.desc()).all()
        logger.info(f"Found {len(recent_baths)} recent baths")
        
        # Get user's language preference
        settings = db.query(Settings).filter(Settings.profile_id == profile.server_id).first()
        language = settings.language if settings else 'zh'  # Default to Chinese if no settings
        
        # Format activities into a readable string
        activities_summary = []
        
        # Feeding summary
        if recent_feedings:
            feeding_types = {}
            for feeding in recent_feedings:
                feeding_types[feeding.type] = feeding_types.get(feeding.type, 0) + 1
            
            if language == 'en':
                feeding_summary = ", ".join([f"{count} {type} feedings" for type, count in feeding_types.items()])
                activities_summary.append(f"Feeding in the last 3 days: {feeding_summary}")
            else:
                feeding_summary = "，".join([f"{count}次{type}" for type, count in feeding_types.items()])
                activities_summary.append(f"最近3天喂养情况：{feeding_summary}")
            logger.info(f"Feeding summary: {feeding_summary}")
        
        # Diaper summary
        if recent_diapers:
            diaper_types = {}
            for diaper in recent_diapers:
                diaper_types[diaper.type] = diaper_types.get(diaper.type, 0) + 1
            
            if language == 'en':
                diaper_summary = ", ".join([f"{count} {type} diapers" for type, count in diaper_types.items()])
                activities_summary.append(f"Diaper changes in the last 3 days: {diaper_summary}")
            else:
                diaper_summary = "，".join([f"{count}次{type}" for type, count in diaper_types.items()])
                activities_summary.append(f"最近3天尿布情况：{diaper_summary}")
            logger.info(f"Diaper summary: {diaper_summary}")
        
        # Sleep summary
        if recent_sleeps:
            total_sleep_minutes = sum(
                (sleep.end_time - sleep.start_time).total_seconds() / 60 
                for sleep in recent_sleeps 
                if sleep.end_time
            )
            avg_sleep_hours = round(total_sleep_minutes / len(recent_sleeps) / 60, 1)
            
            if language == 'en':
                activities_summary.append(f"Average sleep duration in the last 3 days: {avg_sleep_hours} hours")
            else:
                activities_summary.append(f"最近3天平均睡眠时长：{avg_sleep_hours}小时")
            logger.info(f"Sleep summary: {avg_sleep_hours} hours average")
        
        # Bath summary
        if recent_baths:
            if language == 'en':
                activities_summary.append(f"Number of baths in the last 3 days: {len(recent_baths)}")
            else:
                activities_summary.append(f"最近3天洗澡次数：{len(recent_baths)}次")
            logger.info(f"Bath summary: {len(recent_baths)} baths")
        
        # Build initial context dictionary
        context = {
            "name": profile.name,
            "age_months": age_months,
            "age_days": age_days,
            "birth_date": profile.birth_date.isoformat() if profile.birth_date else None,
            "height": profile.height if recent_growth else None,
            "weight": profile.weight if recent_growth else None,
            "gender": profile.gender,
            "created_at": profile.created_at.isoformat() if profile.created_at else None,
            "activities_summary": activities_summary,
            "recent_feedings": len(recent_feedings),
            "recent_diapers": len(recent_diapers),
            "recent_sleeps": len(recent_sleeps),
            "recent_baths": len(recent_baths),
            "latest_growth": {
                "height": recent_growth.height if recent_growth else None,
                "weight": recent_growth.weight if recent_growth else None,
                "head_circumference": recent_growth.head_circumference if recent_growth else None,
                "timestamp": recent_growth.created_at.isoformat() if recent_growth and recent_growth.created_at else None,
                "notes": recent_growth.notes if recent_growth else None
            } if recent_growth else None
        }
        
        # Format measurements text
        measurements = []
        if context.get('height'):
            if language == 'en':
                measurements.append(f"Height {context['height']}cm")
            else:
                measurements.append(f"身高 {context['height']}cm")
        if context.get('weight'):
            if language == 'en':
                measurements.append(f"Weight {context['weight']}kg")
            else:
                measurements.append(f"体重 {context['weight']}kg")
        measurements_text = ", ".join(measurements) if language == 'en' else "，".join(measurements)
        if not measurements:
            measurements_text = "No records" if language == 'en' else "暂无记录"
        
        # Format growth information
        growth_info = []
        if context.get('latest_growth'):
            growth = context['latest_growth']
            if growth.get('height'):
                if language == 'en':
                    growth_info.append(f"Latest height {growth['height']}cm")
                else:
                    growth_info.append(f"最新身高 {growth['height']}cm")
            if growth.get('weight'):
                if language == 'en':
                    growth_info.append(f"Latest weight {growth['weight']}kg")
                else:
                    growth_info.append(f"最新体重 {growth['weight']}kg")
            if growth.get('head_circumference'):
                if language == 'en':
                    growth_info.append(f"Latest head circumference {growth['head_circumference']}cm")
                else:
                    growth_info.append(f"最新头围 {growth['head_circumference']}cm")
            if growth.get('notes'):
                if language == 'en':
                    growth_info.append(f"Notes: {growth['notes']}")
                else:
                    growth_info.append(f"备注：{growth['notes']}")
        growth_text = ", ".join(growth_info) if language == 'en' else "，".join(growth_info)
        if not growth_info:
            growth_text = "No records" if language == 'en' else "暂无记录"
        
        # Add formatted measurements and growth info to context
        context.update({
            "measurements_text": measurements_text,
            "growth_text": growth_text
        })
        
        logger.info("Generated context:")
        logger.info(json.dumps(context, indent=2, ensure_ascii=False))
        
        return context
        
    except Exception as e:
        logger.error(f"Error building baby context: {str(e)}")
        logger.error(traceback.format_exc())
        # Return fallback context with initialized age values
        return {
            "name": profile.name if profile else "Unknown",
            "age_months": age_months,
            "age_days": age_days,
            "birth_date": None,
            "height": None,
            "weight": None,
            "gender": None,
            "created_at": None,
            "activities_summary": [],
            "recent_feedings": 0,
            "recent_diapers": 0,
            "recent_sleeps": 0,
            "recent_baths": 0,
            "latest_growth": None,
            "measurements_text": "No records" if language == 'en' else "暂无记录",
            "growth_text": "No records" if language == 'en' else "暂无记录"
        }

def _filter_inappropriate_content(videos):
    """Additional filtering beyond YouTube's safeSearch"""
    filtered = []
    
    # Define comprehensive blacklist categories
    blacklist_categories = {
        "political": [
            "政治", "选举", "政党", "政府", "抗议", "示威", "革命", "战争", "冲突",
            "争议", "意识形态", "政权", "领导人", "总统", "总理", "部长",
            "政策", "法律", "法规", "权利", "自由", "民主", "专制", "独裁"
        ],
        "inappropriate": [
            "成人", "暴力", "恐怖", "赌博", "色情", "血腥", "残忍", "虐待",
            "自杀", "自残", "毒品", "酒精", "香烟", "烟草", "酗酒", "吸毒",
            "犯罪", "违法", "非法", "黑社会", "帮派", "武器", "枪支", "弹药", "处女"
        ],
        "sensitive": [
            "死亡", "尸体", "葬礼", "坟墓", "鬼魂", "灵异", "诅咒", "邪教",
            "迷信", "占卜", "算命", "风水", "巫术", "魔法", "诅咒", "降头",
            "可怕", "恐怖", "惊悚", "恶心", "恐怖片", "恐怖电影", "恐怖故事", "恐怖小说", "恐怖视频",
            "黑幕", "阴谋", "阴险", "阴暗", "阴谋论", "阴谋论者", "阴谋论者", "阴谋论者", "阴谋论者", "阴谋论者"
        ],
        "unprofessional": [
            "恶搞", "整蛊", "恶趣味", "低俗", "粗俗", "下流",
            "谩骂", "辱骂", "诽谤", "造谣", "谣言", "虚假", "诈骗", "骗局"
        ]
    }
    
    # Combine all blacklist terms
    all_blacklist_terms = []
    for category in blacklist_categories.values():
        all_blacklist_terms.extend(category)
    
    logger.info(f"Starting content filtering with {len(all_blacklist_terms)} blacklist terms")
    
    for video in videos:
        try:
            # Skip videos with invalid format
            if not isinstance(video, dict):
                logger.warning(f"Unexpected video format: {type(video)}")
                continue
                
            # Get title and description safely with defaults
            title = str(video.get("title", "")).lower()
            description = str(video.get("description", "")).lower()
            
            # Check for blacklisted terms in title and description
            found_terms = []
            for term in all_blacklist_terms:
                if term in title or term in description:
                    found_terms.append(term)
            
            if found_terms:
                logger.info(f"Filtering out video with inappropriate content: {title}")
                logger.info(f"Found blacklisted terms: {', '.join(found_terms)}")
                continue
            
            # Additional checks for video metadata
            if video.get("channel", "").lower() in ["新闻", "政治", "时政", "军事"]:
                logger.info(f"Filtering out video from inappropriate channel: {video.get('channel')}")
                continue
                
            # Check video duration
            duration = video.get("duration", "")
            if duration:
                # Parse duration in format "1h 17m 30s" or "10m 30s" or "30s"
                hours = 0
                minutes = 0
                seconds = 0
                
                # Handle hours
                if "h" in duration:
                    hours = int(duration.split("h")[0])
                    duration = duration.split("h")[1].strip()
                
                # Handle minutes
                if "m" in duration:
                    minutes = int(duration.split("m")[0])
                    duration = duration.split("m")[1].strip()
                
                # Handle seconds
                if "s" in duration:
                    seconds = int(duration.split("s")[0])
                
                total_seconds = hours * 3600 + minutes * 60 + seconds
                
                # Skip videos shorter than 8 minutes
                if total_seconds < 480:
                    #logger.info(f"Filtering out short video: {duration}")
                    continue
                
                # Skip videos longer than 1 hour
                if total_seconds > 3600:
                    #logger.info(f"Filtering out long video: {duration}")
                    continue
            
            # Check view count (skip videos with very low views)
            view_count = video.get("view_count", "0")
            if isinstance(view_count, str):
                view_count = view_count.replace(",", "")
            try:
                if int(view_count) < 1000:  # Skip videos with less than 1,000 views
                    logger.info(f"Filtering out low-view video: {view_count} views")
                    continue
            except (ValueError, TypeError):
                pass
            
            # If video passes all filters, add it to filtered list
            filtered.append(video)
            logger.debug(f"Video passed filters: {title}")
            
        except Exception as e:
            logger.error(f"Error filtering video: {str(e)}")
            logger.error(traceback.format_exc())
            continue
    
    logger.info(f"Filtered {len(videos) - len(filtered)} videos, {len(filtered)} videos remaining")
    return filtered

def format_videos_as_cards(videos, profile):
    """Format videos as feed cards with age-specific titles."""
    cards = []
    
    # Get baby's age in months
    baby_age_months = calculate_age_in_months(profile.birth_date)
    
    # Create age-specific title
    if baby_age_months == 0:
        age_title = "新生儿精选"
    elif baby_age_months < 12:
        age_title = f"{baby_age_months}月龄精选"
    else:
        years = baby_age_months // 12
        age_title = f"{years}岁宝宝精选"
    
    for video in videos:
        card = {
            "type": "parent_watch",
            "title": video["title"],  # Use the dynamic age-specific title
            "content": video["title"],
            "summary": video["description"],
            "thumbnail": video["thumbnail"],
            "media_link": video["url"],
            "embed_url": video["embed_url"],
            "channel": video["channel"],
            "duration": video["duration"],
            "view_count": video["view_count"],
            "cta": "观看视频"
        }
        cards.append(card)
    
    return cards

def calculate_age_in_months(birth_date):
    """Calculate age in months from birth date."""
    from datetime import datetime
    today = datetime.now().date()
    
    # Calculate years and months
    years = today.year - birth_date.year
    months = today.month - birth_date.month
    
    # Adjust if birth day hasn't occurred yet this month
    if today.day < birth_date.day:
        months -= 1
    
    # Convert to total months
    total_months = years * 12 + months
    
    # Ensure we don't return negative values
    return max(0, total_months) 
