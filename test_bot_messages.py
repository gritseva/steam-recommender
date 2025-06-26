#!/usr/bin/env python3
"""
Test Messages for SteamRecs Bot

This file contains various test messages to verify all bot functionality.
You can copy and paste these messages into your Telegram bot to test different features.

Usage:
1. Start your bot with: python main.py
2. Open Telegram and find your bot
3. Copy and paste these messages one by one to test different features
"""

# ============================================================================
# BASIC COMMANDS (Slash commands)
# ============================================================================

BASIC_COMMANDS = [
    "/start",
    "/help",
    "/recommend",
    "/feedback",
    "/price_tracker",
    "/reminder",
    "/compare",
    "/video",
    "/filter",
    "/additionalinfo"
]

# ============================================================================
# RECOMMENDATION TESTS
# ============================================================================

RECOMMENDATION_MESSAGES = [
    # Basic recommendation requests
    "I want game recommendations similar to Minecraft",
    "Can you recommend games like Portal 2?",
    "I'm looking for games similar to The Witcher 3",
    "Recommend me some games like Skyrim",
    "I love Counter-Strike, what else should I play?",

    # More specific requests
    "I want multiplayer games like Among Us",
    "Recommend me some RPG games",
    "I'm looking for strategy games",
    "Show me some indie games",
    "I want games with good story like Life is Strange",

    # With preferences
    "I like action games, recommend something",
    "I prefer games under $20",
    "I want free games to play",
    "Recommend me some casual games",
    "I'm looking for challenging games"
]

# ============================================================================
# STEAM PROFILE TESTS
# ============================================================================

STEAM_PROFILE_MESSAGES = [
    # Steam ID formats
    "My Steam ID is 76561198129676583",
    "Here's my Steam profile: https://steamcommunity.com/id/example",
    "Steam ID: 76561198000000000",

    # Greetings (should trigger profile handler)
    "Hello!",
    "Hi there",
    "Hey bot",
    "Good morning",
    "Good evening",

    # Profile analysis requests
    "Can you analyze my gaming profile?",
    "What games do I play the most?",
    "Tell me about my gaming preferences"
]

# ============================================================================
# PRICE TRACKING TESTS
# ============================================================================

PRICE_TRACKING_MESSAGES = [
    # Price tracking requests
    "Track the price of Cyberpunk 2077",
    "Monitor the price for Red Dead Redemption 2",
    "I want to track the price of Minecraft",
    "Keep an eye on the price of Portal 2",
    "Track prices for The Witcher 3",

    # Multiple games
    "Track prices for Minecraft and Portal 2",
    "Monitor the price of Cyberpunk 2077 and Red Dead Redemption 2"
]

# ============================================================================
# REMINDER TESTS
# ============================================================================

REMINDER_MESSAGES = [
    # Time-based reminders
    "Remind me to play Minecraft at 8 PM",
    "Set a reminder for Portal 2 tomorrow at 2 PM",
    "Remind me to play The Witcher 3 on Friday at 7 PM",
    "Set a reminder for Cyberpunk 2077 at 9 PM today",
    "Remind me to play Red Dead Redemption 2 next Monday at 6 PM",

    # Different time formats
    "Remind me to play Minecraft in 2 hours",
    "Set a reminder for Portal 2 in 30 minutes",
    "Remind me to play The Witcher 3 at midnight"
]

# ============================================================================
# GAME COMPARISON TESTS
# ============================================================================

COMPARISON_MESSAGES = [
    # Two game comparisons
    "Compare Minecraft and Terraria",
    "What's the difference between Portal and Portal 2?",
    "Compare The Witcher 3 and Skyrim",
    "Which is better: Cyberpunk 2077 or Red Dead Redemption 2?",
    "Compare Counter-Strike and Valorant",

    # More comparisons
    "Minecraft vs Terraria - which should I buy?",
    "Portal 2 vs The Talos Principle",
    "The Witcher 3 vs Dragon Age: Inquisition"
]

# ============================================================================
# VIDEO SEARCH TESTS
# ============================================================================

VIDEO_SEARCH_MESSAGES = [
    # Video requests
    "Show me a video of Minecraft gameplay",
    "I want to see a Portal 2 walkthrough",
    "Find me a review of The Witcher 3",
    "Show me Cyberpunk 2077 gameplay",
    "I want to see Red Dead Redemption 2 trailers",

    # Specific video types
    "Show me a tutorial for Minecraft",
    "Find me a speedrun of Portal 2",
    "I want to see The Witcher 3 review"
]

# ============================================================================
# FEEDBACK TESTS
# ============================================================================

FEEDBACK_MESSAGES = [
    # Positive feedback
    "I love the recommendations you gave me!",
    "Great job on the game suggestions",
    "The price tracking feature is really useful",
    "Thanks for the helpful recommendations",

    # Negative feedback
    "I didn't like the games you recommended",
    "The recommendations weren't very good",
    "Can you suggest different types of games?",
    "I want more variety in recommendations"
]

# ============================================================================
# CONTENT FILTER TESTS
# ============================================================================

CONTENT_FILTER_MESSAGES = [
    # Content preferences
    "I don't want to see violent games",
    "Filter out horror games",
    "I prefer family-friendly games",
    "Don't show me games with mature content",
    "I want to avoid games with microtransactions",

    # Genre preferences
    "I don't like sports games",
    "Filter out racing games",
    "I prefer not to see puzzle games"
]

# ============================================================================
# OPINION REQUEST TESTS
# ============================================================================

OPINION_MESSAGES = [
    # Opinion requests
    "What do you think about Minecraft?",
    "Give me your opinion on Portal 2",
    "What's your take on The Witcher 3?",
    "Tell me what you think about Cyberpunk 2077",
    "What's your opinion on Red Dead Redemption 2?"
]

# ============================================================================
# TOP GAMES TESTS
# ============================================================================

TOP_GAMES_MESSAGES = [
    # Genre requests
    "What are the top RPG games?",
    "Show me the best action games",
    "What are the top strategy games?",
    "Best indie games?",
    "Top multiplayer games",

    # Specific requests
    "What are the best games in the adventure genre?",
    "Show me the top survival games",
    "Best puzzle games?"
]

# ============================================================================
# TRANSLATION TESTS
# ============================================================================

TRANSLATION_MESSAGES = [
    # Non-English messages (if you want to test translation)
    "Hola, quiero recomendaciones de juegos",
    "Bonjour, je veux des recommandations de jeux",
    "Ciao, voglio raccomandazioni di giochi",
    "Hallo, ich möchte Spieleempfehlungen"
]

# ============================================================================
# OUT OF CONTEXT TESTS
# ============================================================================

OUT_OF_CONTEXT_MESSAGES = [
    # Non-gaming topics
    "What's the weather like today?",
    "Tell me a joke",
    "What's 2+2?",
    "How do I cook pasta?",
    "What's the capital of France?",

    # General questions
    "Who are you?",
    "What can you do?",
    "Are you a real person?"
]

# ============================================================================
# COMPREHENSIVE TEST SEQUENCE
# ============================================================================


def print_test_sequence():
    """Print a comprehensive test sequence for the bot"""

    print("=" * 60)
    print("STEAMRECS BOT TEST SEQUENCE")
    print("=" * 60)
    print("Copy and paste these messages into your Telegram bot to test functionality.")
    print("Test them in order for best results.\n")

    test_categories = [
        ("BASIC COMMANDS", BASIC_COMMANDS),
        ("STEAM PROFILE & GREETINGS", STEAM_PROFILE_MESSAGES),
        ("RECOMMENDATIONS", RECOMMENDATION_MESSAGES),
        ("PRICE TRACKING", PRICE_TRACKING_MESSAGES),
        ("REMINDERS", REMINDER_MESSAGES),
        ("GAME COMPARISONS", COMPARISON_MESSAGES),
        ("VIDEO SEARCH", VIDEO_SEARCH_MESSAGES),
        ("FEEDBACK", FEEDBACK_MESSAGES),
        ("CONTENT FILTERS", CONTENT_FILTER_MESSAGES),
        ("OPINION REQUESTS", OPINION_MESSAGES),
        ("TOP GAMES", TOP_GAMES_MESSAGES),
        ("OUT OF CONTEXT", OUT_OF_CONTEXT_MESSAGES)
    ]

    for category, messages in test_categories:
        print(f"\n{category}")
        print("-" * len(category))
        for i, message in enumerate(messages, 1):
            print(f"{i}. {message}")

    print("\n" + "=" * 60)
    print("TESTING INSTRUCTIONS:")
    print("1. Start your bot: python main.py")
    print("2. Open Telegram and find your bot")
    print("3. Copy and paste messages one by one")
    print("4. Check that the bot responds appropriately")
    print("5. Look for any error messages in the console")
    print("=" * 60)


def print_quick_test():
    """Print a quick test with essential messages"""

    print("=" * 60)
    print("QUICK TEST - ESSENTIAL MESSAGES")
    print("=" * 60)

    quick_tests = [
        "Hello!",
        "I want game recommendations similar to Minecraft",
        "My Steam ID is 76561198129676583",
        "Compare Minecraft and Terraria",
        "Track the price of Cyberpunk 2077",
        "Remind me to play Portal 2 at 8 PM",
        "Show me a video of Minecraft gameplay",
        "I love the recommendations you gave me!",
        "What do you think about The Witcher 3?",
        "What are the top RPG games?"
    ]

    for i, message in enumerate(quick_tests, 1):
        print(f"{i}. {message}")

    print("\nTest these 10 messages to verify core functionality.")


if __name__ == "__main__":
    print("Choose test type:")
    print("1. Quick test (10 essential messages)")
    print("2. Comprehensive test (all messages)")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        print_quick_test()
    else:
        print_test_sequence()
