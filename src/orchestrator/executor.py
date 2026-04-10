from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Task, TaskState, UnsupportedOperationError, InvalidRequestError
from a2a.utils.errors import ServerError
from a2a.utils import new_agent_text_message, new_task

from agent import MLAgent

TERMINAL_STATES = {TaskState.completed, TaskState.canceled, TaskState.failed, TaskState.rejected}


class Executor(AgentExecutor):
    def __init__(self):
        self.agents: dict[str, MLAgent] = {}

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        msg = context.message
        if not msg:
            raise ServerError(error=InvalidRequestError(message="Missing message"))

        task = context.current_task
        if task and task.status.state in TERMINAL_STATES:
            raise ServerError(error=InvalidRequestError(message=f"Task {task.id} already in terminal state"))

        if not task:
            task = new_task(msg)
            await event_queue.enqueue_event(task)

        context_id = task.context_id
        agent = self.agents.get(context_id)
        if not agent:
            agent = MLAgent()
            self.agents[context_id] = agent

        updater = TaskUpdater(event_queue, task.id, context_id)
        await updater.start_work()
        try:
            await agent.run(msg, updater)
            if not updater._terminal_state_reached:
                await updater.complete()
        except Exception as e:
            print(f"Agent error: {e}")
            await updater.failed(new_agent_text_message(f"Agent error: {e}"))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError())
