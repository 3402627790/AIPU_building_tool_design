import random
from .logger import *
from abc import ABC, abstractmethod


class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_possible_actions())

    def best_child(self, exploration_weight=1.0):
        return max(
            self.children,
            key=lambda child: child.reward / (child.visits or 1)
            + exploration_weight * ((2 * (self.visits or 1)) ** 0.5 / (child.visits or 1)),
        )


class GameState(ABC):
    @abstractmethod
    def get_possible_actions(self):
        pass

    @abstractmethod
    def is_terminal(self):
        pass

    @abstractmethod
    def get_reward(self):
        pass

    @abstractmethod
    def take_action(self, action):
        pass


class MCTS:
    def __init__(self, initial_state):
        self.root = Node(initial_state)

    def search(self):
        # self.root.state.take_action(0)
        logger.info("select")
        nodes = self.select()
        for node in nodes:
            logger.info("simulate")
            self.dfs_simulate(node)

    def select(self):
        nodes = []
        while not self.root.is_fully_expanded():
            logger.info("not fully expanded because children: " + str(len(self.root.children)) + " actions: " + str(len(self.root.state.get_possible_actions())))
            nodes.append(self.expand(self.root))    
        return nodes

    def expand(self, node):
        logger.info("expand")
        possible_actions = node.state.get_possible_actions()
        # actions_not_taken = [
        #     action for action in possible_actions if action not in [child.state.last_action for child in node.children]
        # ]
        executed_actions = {child.state.action for child in node.children}
        logger.info("excuted_actions: " + str(len(executed_actions)))
        actions_not_taken = [action for action in possible_actions if action not in executed_actions]
        # action = random.choice(actions_not_taken)
        if len(actions_not_taken) == 0:
            logger.info("possible_actions: " + str(possible_actions))
            logger.info("executed_actions: " + str(executed_actions))


        action = actions_not_taken[0]
        logger.info("get choice " + str(action))
        new_state = node.state.take_action(action)
        new_node = Node(new_state, parent=node)
        node.children.append(new_node)
        return new_node


    def dfs_simulate(self, node, depth=0, visited=None):
        if visited is None:
            visited = set()

        if node in visited:
            logger.info("Node already visited, stop recursion")
            return

        visited.add(node)
        logger.info("current depth " + str(depth))


        possible_actions = node.state.get_possible_actions()

        if node.state.is_terminal():
            return

        for action in possible_actions:
            new_state = node.state.take_action(action)
            new_node = Node(new_state, parent=node)
            logger.info("dfs_simulate new_node " + str(new_node) + " from node " + str(node))
            node.children.append(new_node)
            self.dfs_simulate(new_node, depth + 1, visited)

    def get_best_state(self):
        best_node = self.dfs_find_min_recursive(self.root)
        if best_node is None:
            logger.error("error")
        if best_node.state is None:
            logger.error("error")
        logger.info("best node cycle " + str(best_node.state.get_reward()))
        return best_node.state


    def dfs_find_min_recursive(self, node, min_node=None):
        logger.info("node cycle " + str(node.state.get_reward()) + " strategy " + node.state.strategy_file)
        if min_node is not None:
            logger.info("min cycle " + str(node.state.get_reward()))
        if node is None:
            logger.error("error")
        if node.state is None:
            logger.error("error")
        if node.state.get_reward() > 0:
            if min_node is None or node.state.get_reward() < min_node.state.get_reward():
                min_node = node
        logger.info("child num " + str(len(node.children)))

        for child in node.children:
            # if not child.state.is_terminal():
            min_node = self.dfs_find_min_recursive(child, min_node)
        return min_node

    def get_state_num(self):
        return 1 + self.get_children_num(self.root)
        
    def get_children_num(self,node):
        cnt = 0
        for child in node.children:
            cnt = cnt + 1 + self.get_children_num(child)
        return cnt
