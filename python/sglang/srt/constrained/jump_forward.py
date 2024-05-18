import interegular

from sglang.srt.constrained import FSMInfo, disk_cache, make_deterministic_fsm
from sglang.srt.constrained.base_cache import BaseCache

IP_REGEX = r"((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)"


class JumpForwardMap:
    def __init__(self, regex_string):
        @disk_cache()
        def _init_state_to_jump_forward(regex_string):
            regex_pattern = interegular.parse_pattern(regex_string)
            regex_fsm, _ = make_deterministic_fsm(regex_pattern.to_fsm().reduce())

            fsm_info: FSMInfo = regex_fsm.fsm_info

            symbol_to_id = fsm_info.alphabet_symbol_mapping
            id_to_symbol = {}
            for symbol, id_ in symbol_to_id.items():
                id_to_symbol.setdefault(id_, []).append(symbol)

            transitions = fsm_info.transitions
            dirty_states = set()
            state_to_jump_forward = {}

            for (state, id_), next_state in transitions.items():
                if state in dirty_states:
                    continue
                if state in state_to_jump_forward:
                    dirty_states.add(state)
                    del state_to_jump_forward[state]
                    continue
                if len(id_to_symbol[id_]) > 1:
                    dirty_states.add(state)
                    continue

                state_to_jump_forward[state] = (id_to_symbol[id_][0], next_state)

            return state_to_jump_forward

        self.state_to_jump_forward = _init_state_to_jump_forward(regex_string)

    def valid_states(self):
        return self.state_to_jump_forward.keys()

    def jump_forward(self, state):
        if state not in self.state_to_jump_forward:
            return None

        jump_forward_str = ""
        next_state = None
        while state in self.state_to_jump_forward:
            symbol, next_state = self.state_to_jump_forward[state]
            jump_forward_str += symbol
            state = next_state
        return jump_forward_str, next_state


class JumpForwardCache(BaseCache):
    def __init__(self):
        super().__init__()

    def init_value(self, regex):
        return JumpForwardMap(regex)


def test_main():
    regex_string = r"The google's DNS sever address is " + IP_REGEX
    jump_forward_map = JumpForwardMap(regex_string)
    for state in jump_forward_map.valid_states():
        print(state, f'"{jump_forward_map.jump_forward(state)}"')


if __name__ == "__main__":
    test_main()
