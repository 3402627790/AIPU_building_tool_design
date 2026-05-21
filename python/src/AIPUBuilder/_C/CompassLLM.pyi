from AIPUBuilder.cruntimc import CompassModelGenerationStates_t


class CompassLLMConfig_Pyd:

    def show(): - -> str
    '''
    show the configuration for compass LLM
    '''
    ...


def LLM_init(config): - -> int

'''
    Initialize a LLM runtime instance, and returns its sess id
    '''
...


def LLM_destroy(sess_id): - -> CompassModelGenerationStates_t

'''
    Deinitialize a LLM runtime
    '''
...


def LLM_run(sess_id, text): - -> CompassModelGenerationStates_t

'''
    Start a query
    '''
...


def LLM_setSystemPrompt(sess_id, text): - -> CompassModelGenerationStates_t

'''
    Set system prompt for LLM
    '''
...
