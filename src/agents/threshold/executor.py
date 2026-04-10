from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState, UnsupportedOperationError, InvalidRequestError
from a2a.utils.errors import ServerError
from a2a.utils import new_agent_text_message, new_task

from agent import ThresholdOptimizerAgent

TERMINAL_STATES = {TaskState.completed, TaskState.canceled, TaskState.failed, TaskState.rejected}


class Executor(AgentExecutor):
    def __init__(self):
        self.agent = ThresholdOptimizerAgent()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        msg = context.message
        if not msg:
            raise ServerError(error=InvalidRequestError(message="Missing message"))

        task = context.current_task
        if task and task.status.state in TERMINAL_STATES:
            raise ServerError(error=InvalidRequestError(message="Task already in terminal state"))

        if not task:
            task = new_task(msg)
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.start_work()
        try:
            await self.agent.run(msg, updater)
            if not updater._terminal_state_reached:
                await updater.complete()
        except Exception as e:
            print(f"[Thresh] Executor error: {e}")
            await updater.failed(new_agent_text_message(f"Threshold Optimizer agent failed: {e}"))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError())
