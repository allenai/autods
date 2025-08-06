from autogen import GroupChat, Agent
from typing import Optional
import json


class SpeakerSelector:
    def __init__(self):
        self.code_failure_count = 0
        self.experiment_revision_count = 0

    def select_next_speaker(self, last_speaker: Agent, groupchat: GroupChat) -> Optional[Agent]:
        """Define a customized speaker selection function for the data exploration workflow.
        
        Args:
            last_speaker: The previous speaker in the conversation
            groupchat: The GroupChat instance containing conversation history
            
        Returns:
            The next agent to speak or None to end the conversation
        """
        messages = groupchat.messages

        if last_speaker.name == "user_proxy":
            return groupchat.agent_by_name("experiment_programmer")

        elif last_speaker.name == "experiment_programmer":
            return groupchat.agent_by_name("code_executor")

        elif last_speaker.name == "code_executor":
            return groupchat.agent_by_name("experiment_code_analyst")

        elif last_speaker.name == "experiment_code_analyst":
            # Check if experiment failed based on structured response
            content = messages[-1].get("content", "")
            try:
                response = json.loads(content)
            except json.JSONDecodeError:
                # If JSON parsing fails, treat it as an error
                response = {"success": False, "analysis": "Error parsing response"}
            if not response.get("success", False) and self.code_failure_count < 6:
                self.code_failure_count += 1
                return groupchat.agent_by_name("experiment_programmer")
            else:
                self.code_failure_count = 0
                return groupchat.agent_by_name("experiment_reviewer")

        elif last_speaker.name == "experiment_reviewer":
            content = messages[-1].get("content", "")
            try:
                response = json.loads(content)
            except json.JSONDecodeError:
                response = {"success": False, "feedback": "Error parsing reviewer response"}
            if not response.get("success", True) and self.experiment_revision_count < 1:
                self.experiment_revision_count += 1
                return groupchat.agent_by_name("experiment_reviser")
            else:
                self.experiment_revision_count = 0
                return groupchat.agent_by_name("experiment_generator")

        if last_speaker.name == "experiment_reviser":
            return groupchat.agent_by_name("experiment_programmer")

        elif last_speaker.name == "experiment_generator":
            return None

        return None
