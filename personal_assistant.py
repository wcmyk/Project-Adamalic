"""
AI Personal Assistant - $2-3/month
Handles scheduling, calendar, emails, and more

Setup:
1. pip install anthropic requests google-auth-httplib2 google-auth-oauthlib google-api-python-client
2. Set environment variables for API keys
3. python personal_assistant.py
"""

import os
from datetime import datetime, timedelta
from anthropic import Anthropic


class PersonalAssistant:
    """AI-powered personal assistant using Claude API."""

    def __init__(self):
        self.claude = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.system_prompt = """You are a helpful personal assistant.
        You help with scheduling, calendar management, ride booking, and daily tasks.
        Always be concise, actionable, and proactive."""

    # === SMART ROUTING ===

    def handle_request(self, request: str):
        """Main entry point - routes to appropriate handler."""

        request_lower = request.lower()

        # Route to specific handlers
        if "uber" in request_lower or "lyft" in request_lower or "ride" in request_lower:
            return self.schedule_ride(request)

        elif "calendar" in request_lower or "schedule" in request_lower or "meeting" in request_lower:
            return self.manage_calendar(request)

        elif "email" in request_lower or "draft" in request_lower or "reply" in request_lower:
            return self.handle_email(request)

        elif "remind" in request_lower or "reminder" in request_lower:
            return self.set_reminder(request)

        else:
            # General AI assistance
            return self.general_assistance(request)

    # === RIDE SCHEDULING ===

    def schedule_ride(self, request: str):
        """Schedule Uber/Lyft ride using AI to parse request."""

        # Use AI to extract ride details
        response = self.claude.messages.create(
            model="claude-sonnet-4",
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": f"""Extract ride details from this request and return JSON:
                "{request}"

                Return format:
                {{
                    "pickup": "address or 'current location'",
                    "destination": "address",
                    "time": "ISO datetime or 'now'",
                    "service": "uber or lyft"
                }}"""
            }]
        )

        # Parse AI response
        import json
        details = json.loads(response.content[0].text)

        # Call ride service API (implement your preferred service)
        result = self._call_uber_api(details)

        return f"✅ Ride scheduled:\n{details['pickup']} → {details['destination']}\nTime: {details['time']}\n{result}"

    def _call_uber_api(self, details: dict):
        """Call Uber API to schedule ride."""
        # Placeholder - implement with actual Uber API
        # https://developer.uber.com/docs/riders/ride-requests/tutorials/api/python

        return "Uber API integration needed - add your credentials"

    # === CALENDAR MANAGEMENT ===

    def manage_calendar(self, request: str):
        """Add/modify calendar events using AI."""

        response = self.claude.messages.create(
            model="claude-sonnet-4",
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": f"""Extract calendar event details:
                "{request}"

                Return JSON:
                {{
                    "action": "add/remove/modify",
                    "title": "event title",
                    "start_time": "ISO datetime",
                    "end_time": "ISO datetime",
                    "description": "optional description"
                }}"""
            }]
        )

        import json
        event = json.loads(response.content[0].text)

        # Add to Google Calendar
        result = self._add_to_google_calendar(event)

        return f"✅ Calendar updated:\n{event['title']} on {event['start_time']}\n{result}"

    def _add_to_google_calendar(self, event: dict):
        """Add event to Google Calendar."""
        # Placeholder - implement Google Calendar API
        # https://developers.google.com/calendar/api/quickstart/python

        try:
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build

            # Initialize calendar service
            service = build('calendar', 'v3', credentials=self._get_google_creds())

            # Create event
            event_data = {
                'summary': event['title'],
                'start': {'dateTime': event['start_time']},
                'end': {'dateTime': event['end_time']},
                'description': event.get('description', ''),
            }

            created = service.events().insert(calendarId='primary', body=event_data).execute()
            return f"Event created: {created.get('htmlLink')}"

        except Exception as e:
            return f"Calendar API setup needed: {e}"

    def _get_google_creds(self):
        """Get Google Calendar credentials."""
        # Implement OAuth2 flow
        # Follow: https://developers.google.com/calendar/api/quickstart/python
        pass

    # === EMAIL HANDLING ===

    def handle_email(self, request: str):
        """Draft or send emails using AI."""

        response = self.claude.messages.create(
            model="claude-sonnet-4",
            max_tokens=1000,
            messages=[{
                "role": "user",
                "content": f"""Draft an email based on this request:
                "{request}"

                Return ONLY the email content (subject and body), formatted professionally."""
            }]
        )

        email_draft = response.content[0].text

        return f"📧 Email draft:\n\n{email_draft}\n\n(Send via your email client or integrate Gmail API)"

    # === REMINDERS ===

    def set_reminder(self, request: str):
        """Set a reminder using AI to parse the request."""

        response = self.claude.messages.create(
            model="claude-sonnet-4",
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": f"""Extract reminder details:
                "{request}"

                Return JSON:
                {{
                    "task": "what to remind",
                    "time": "ISO datetime when to remind"
                }}"""
            }]
        )

        import json
        reminder = json.loads(response.content[0].text)

        # Store reminder (implement your storage)
        result = self._store_reminder(reminder)

        return f"⏰ Reminder set:\n{reminder['task']}\nWhen: {reminder['time']}"

    def _store_reminder(self, reminder: dict):
        """Store reminder in database or calendar."""
        # Implement with your preferred storage
        # Options: SQLite, Google Calendar, iOS Reminders API
        pass

    # === GENERAL ASSISTANCE ===

    def general_assistance(self, request: str):
        """Handle general queries with AI."""

        response = self.claude.messages.create(
            model="claude-sonnet-4",
            max_tokens=1000,
            system=self.system_prompt,
            messages=[{
                "role": "user",
                "content": request
            }]
        )

        return response.content[0].text

    # === DAILY ROUTINES ===

    def morning_briefing(self):
        """Generate morning briefing with weather, calendar, news."""

        # Get today's data
        today = datetime.now().strftime("%Y-%m-%d")

        # Fetch data (implement actual API calls)
        weather = self._get_weather()
        calendar_events = self._get_todays_calendar()

        # AI creates briefing
        response = self.claude.messages.create(
            model="claude-sonnet-4",
            max_tokens=800,
            messages=[{
                "role": "user",
                "content": f"""Create a concise morning briefing:

                Date: {today}
                Weather: {weather}
                Today's Schedule: {calendar_events}

                Format as a friendly, actionable summary."""
            }]
        )

        return response.content[0].text

    def _get_weather(self):
        """Get weather data."""
        # Implement with OpenWeatherMap API (free tier)
        return "Sunny, 72°F"

    def _get_todays_calendar(self):
        """Get today's calendar events."""
        # Implement with Google Calendar API
        return "9am: Team meeting, 2pm: Dentist, 5pm: Gym"


# === CLI INTERFACE ===

def main():
    """Interactive CLI for personal assistant."""

    assistant = PersonalAssistant()

    print("🤖 AI Personal Assistant Ready!")
    print("Examples:")
    print("  - 'Schedule Uber to airport tomorrow at 6am'")
    print("  - 'Add dentist appointment next Tuesday 2pm'")
    print("  - 'Draft email declining meeting invite'")
    print("  - 'Remind me to call mom in 2 hours'")
    print("  - 'What's my schedule today?'")
    print("\nType 'quit' to exit\n")

    while True:
        request = input("You: ")

        if request.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not request.strip():
            continue

        try:
            response = assistant.handle_request(request)
            print(f"\n🤖 Assistant: {response}\n")

        except Exception as e:
            print(f"❌ Error: {e}\n")


# === SCHEDULED TASKS ===

def scheduled_tasks():
    """Run scheduled tasks (use with cron or system scheduler)."""

    assistant = PersonalAssistant()

    # Morning briefing
    if datetime.now().hour == 7:
        briefing = assistant.morning_briefing()
        # Send via SMS/notification
        print(f"Morning briefing: {briefing}")


if __name__ == "__main__":
    main()
