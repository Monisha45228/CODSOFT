"""
Task 1: Chatbot with Rule-Based Responses
-------------------------------------------
A simple chatbot that uses if-else logic and regex pattern matching
to recognize user intents and respond appropriately.

Run this file and chat in the terminal. Type 'quit', 'exit', or 'bye'
to end the conversation.
"""

import re
import random
from datetime import datetime


class RuleBasedChatbot:
    def __init__(self, name="ChatBot"):
        self.name = name
        self.user_name = None

        # Each rule = (regex pattern, list of possible responses OR a handler function)
        # Order matters: more specific patterns should come before general ones.
        self.rules = [
            (r"\b(hi|hello|hey|greetings)\b", self._respond_greeting),
            (r"\bmy name is (\w+)", self._respond_set_name),
            (r"\bwhat('?s| is) your name\b", self._respond_bot_name),
            (r"\bhow are you\b", self._respond_how_are_you),
            (r"\bwhat (time|is the time)\b", self._respond_time),
            (r"\bwhat (day|is the date|date)\b", self._respond_date),
            (r"\b(thank(s)?( you)?)\b", self._respond_thanks),
            (r"\b(bye|goodbye|see you|exit|quit)\b", self._respond_bye),
            (r"\bhow (?:do|can) i (.*)\b", self._respond_how_to),
            (r"\b(who are you|what are you)\b", self._respond_identity),
            (r"\b(help|what can you do)\b", self._respond_help),
            (r"\b(\d+)\s*\+\s*(\d+)\b", self._respond_addition),
            (r"\bjoke\b", self._respond_joke),
        ]

        self.fallback_responses = [
            "I'm not sure I understand. Could you rephrase that?",
            "Interesting! Tell me more.",
            "I don't have a rule for that yet — could you try asking differently?",
            "Hmm, I didn't quite catch that. Can you say it another way?",
        ]

    # ---------- Individual response handlers ----------

    def _respond_greeting(self, match, text):
        greetings = ["Hello there!", "Hi! How can I help you today?", "Hey! Nice to see you."]
        return random.choice(greetings)

    def _respond_set_name(self, match, text):
        self.user_name = match.group(1).capitalize()
        return f"Nice to meet you, {self.user_name}!"

    def _respond_bot_name(self, match, text):
        return f"I'm {self.name}, your friendly rule-based chatbot."

    def _respond_how_are_you(self, match, text):
        return "I'm just a program, but I'm running smoothly! How about you?"

    def _respond_time(self, match, text):
        return f"The current time is {datetime.now().strftime('%H:%M:%S')}."

    def _respond_date(self, match, text):
        return f"Today's date is {datetime.now().strftime('%Y-%m-%d')}."

    def _respond_thanks(self, match, text):
        return "You're welcome!"

    def _respond_bye(self, match, text):
        name_part = f", {self.user_name}" if self.user_name else ""
        return f"Goodbye{name_part}! Have a great day. [SESSION_END]"

    def _respond_how_to(self, match, text):
        topic = match.group(1).strip()
        return f"To {topic}, I'd suggest breaking it into small steps and searching official docs or tutorials for details."

    def _respond_identity(self, match, text):
        return "I'm a rule-based chatbot built with pattern matching and if-else logic — no fancy AI here, just clear rules!"

    def _respond_help(self, match, text):
        return ("I can chat about greetings, tell you the time/date, do simple addition (e.g. '2 + 2'), "
                "tell a joke, remember your name, or just talk. Try saying 'hi', 'help', or 'joke'!")

    def _respond_addition(self, match, text):
        a, b = int(match.group(1)), int(match.group(2))
        return f"{a} + {b} = {a + b}"

    def _respond_joke(self, match, text):
        jokes = [
            "Why do programmers prefer dark mode? Because light attracts bugs!",
            "Why did the computer go to the doctor? It caught a virus!",
            "I would tell you a UDP joke, but you might not get it.",
        ]
        return random.choice(jokes)

    # ---------- Core matching logic ----------

    def get_response(self, user_input):
        text = user_input.strip().lower()

        if not text:
            return "Please say something!"

        for pattern, handler in self.rules:
            match = re.search(pattern, text)
            if match:
                return handler(match, text)

        return random.choice(self.fallback_responses)

    def chat(self):
        print(f"{self.name}: Hi! I'm {self.name}. Type 'quit' to exit.\n")
        while True:
            user_input = input("You: ")
            response = self.get_response(user_input)

            if "[SESSION_END]" in response:
                print(f"{self.name}: {response.replace('[SESSION_END]', '').strip()}")
                break

            print(f"{self.name}: {response}")


if __name__ == "__main__":
    bot = RuleBasedChatbot(name="RuleBot")
    bot.chat()
