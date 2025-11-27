# General imports
import random
import constants
import math
import cv2
import numpy as np

# SC2 API imports
from sc2.bot_ai import BotAI  # parent class we inherit from
from sc2.ids.unit_typeid import UnitTypeId
from sc2.data import Result

MAX_GAME_VALUE = 40000.0


class VRBot(BotAI): # inhereits from BotAI (part of BurnySC2)
    def __init__(self, *args,bot_in_box=None, action_in=None, result_out=None, **kwargs, ):
        super().__init__()
        self.action_in = action_in
        self.result_out = result_out
        self.main_base_destroyed = False
        self.last_action_time = 0
        
        self.enemy_minerals_seen = 0
        self.enemy_vespene_seen = 0
        self.enemy_army_seen = 0
        self.enemy_structures_seen = 0
        self.enemy_voidrays_seen = 0
        
        

    async def on_end(self, game_result):
        print("Game over!")
        obs = np.zeros((224, 224, 3), dtype=np.uint8)
        print(f"GAME DURATION: {self.time}")

        # compute final value similar to on_step (but using final score if available)
        if self.state and self.state.score:
            final_value = (
                self.state.score.total_value_units +
                self.state.score.total_value_structures +
                self.state.score.collected_minerals +
                self.state.score.collected_vespene
            )
        else:
            final_value = 0.0

        # estimate opponent final value as neutral (trainer can combine both ends if desired)
        opponent_final_value = MAX_GAME_VALUE - final_value if 'MAX_GAME_VALUE' in globals() else 0.0

        # scale terminal reward smaller to avoid dominating intrinsic signals
        MAX_TERMINAL_REWARD = 100.0  # reduced from 300 to 100
        value_factor = np.clip(final_value / MAX_GAME_VALUE, 0.0, 1.0)

        if str(game_result) == str(Result.Victory):
            reward = MAX_TERMINAL_REWARD * value_factor
        else:
            reward = -MAX_TERMINAL_REWARD * (1.0 - value_factor)

        info = {"self_value": float(final_value), "opponent_value": float(opponent_final_value),
                "heuristic_reward": float(reward)}
        print(f"TERMINAL REWARD: {reward:.2f} | final_value={final_value:.2f} | value_factor={value_factor:.3f}")

        self.result_out.put({"observation": obs, "reward": float(reward), "action": None, "done": True, "truncated": False, "info": info})

#    async def on_end(self,game_result):
#        print ("Game over!")
#        
#        obs = np.zeros((224, 224, 3), dtype=np.uint8)
#
#        print(f"GAME DURATION: {self.time}")
#        
#        # the game is converged to zero sum game
#        if str(game_result) == "Result.Victory":
#            reward = 300
#        else:
#            reward = -300
#
#        self.result_out.put({"observation" : obs, "reward" : reward, "action" : None, "done" : True, "truncated" : False, "info" : {}})
        
        
    
        

    async def on_step(self, iteration): # on_step is a method that is called every step of the game.
        if self.time - self.last_action_time < 0.5:
            return

        self.last_action_time = self.time  

        game_time_minutes = self.time / 60  # self.time is in seconds

        if game_time_minutes > 30:
            print("30 minutes have passed. Ending the game...")
            # Force a game end. Here, we just surrender.
            await self.client.leave()


        #self.action = self.action_in.get()
        try:
            self.action = self.action_in.get_nowait()
        except:
            self.action = 5

        print(f"Iteration: {iteration} | Action: {self.action} | Game time: {self.time}")
        
        if self.action is None:
            print("no action returning.")
            return None
        
        
        # DRL Agent Logic
        # 0 - Expand, Build Assimilators and Workers (Long-Term Economy)
        # 1 - Build Stargate (Military Building)
        # 2 - Build Void Rays (Military Unit)
        # 3 - Attack with all Void Rays (Attacking the enemy to win rewards)
        # 4 - Build Pylon (Supply)
        # 5 - Do nothing 

        try:
            await self.distribute_workers()
            if self.action == 0:
                await self.expand()
                await self.build_assimilators()
                await self.build_workers()
            elif self.action == 1:
                await self.build_gateway()
                await self.build_cybernetics_core()
                await self.build_stargates()
            elif self.action == 2:
                await self.build_void_rays()
            elif self.action == 3:
                await self.attack()
            elif self.action == 4:
                await self.build_pylons()
            else:
                pass
        except Exception as e:
            print(f"Exception - {e}")

        obs = self.visualize_intel()
        reward = self.reward_function()
        
                # after obs = self.visualize_intel(); reward = self.reward_function()

        # --- compute self_value (economy + military proxy)
        # weights: minerals + vespene + 10 * voidrays + 5 * other army units + 20 * structures
        my_minerals = int(self.minerals)
        my_vespene = int(self.vespene)
        my_voidrays = self.units(UnitTypeId.VOIDRAY).amount
        # count other army units generically
        my_army_units = sum([u.amount for u in [
            self.units(UnitTypeId.ZEALOT),
            self.units(UnitTypeId.STALKER),
            self.units(UnitTypeId.ADEPT),
        ]]) if hasattr(self, 'units') else 0
        my_structures = len(self.structures)

        self_value = my_minerals + my_vespene + 10 * my_voidrays + 5 * my_army_units + 20 * my_structures

        # --- estimate opponent_value using visible enemy units/structures (best-effort)
        # if no vision, fall back to a neutral default (half of MAX_GAME_VALUE)
#        enemy_minerals_seen = 0
#        enemy_vespene_seen = 0
#        enemy_voidrays = 0
#        enemy_army_units = 0
#        enemy_structures = 0
        
#        self.compute_enemy_state()
        
                # inside agent.step(env, obs) or self.on_step(state)
        
               # Define once in __init__ or globally
        self.ARMY_UNITS = {
            UnitTypeId.MARINE,
            UnitTypeId.MARAUDER,
            UnitTypeId.ZEALOT,
            UnitTypeId.STALKER,
            UnitTypeId.VOIDRAY,
            UnitTypeId.ZERGLING,
            UnitTypeId.ROACH,
            UnitTypeId.HYDRALISK,
            UnitTypeId.IMMORTAL,
            UnitTypeId.COLOSSUS,
            # Add more if needed
        }
        
        # Inside on_step:
        visible_enemies = self.enemy_units | self.enemy_structures
        print(visible_enemies)
        
        if visible_enemies:
            # Count structures (this attribute exists)
            self.enemy_structures_seen = len([u for u in visible_enemies if u.is_structure])
        
            # Count army units using the type_id set
            self.enemy_army_seen = len([u for u in visible_enemies if u.type_id in self.ARMY_UNITS])
        
            # Specific unit types
            self.enemy_voidrays_seen = self.enemy_units(UnitTypeId.VOIDRAY).amount

        # estimate from visible enemy units
#        for eu in self.all_enemy_units:
#            self.enemy_struct_id = eu.type_id
#            if eu.type_id == UnitTypeId.VOIDRAY:
#                self.enemy_voidrays += 1
#            else:
#                self.enemy_army_units += 1
#
#        for es in self.enemy_structures:
#            self.enemy_structures += 1

        # simple opponent value proxy using only visible info
        opponent_value = (self.enemy_voidrays_seen * 10 + self.enemy_army_seen * 3 +
                          10 * self.enemy_structures_seen + 5)
        # if we have no visible enemy info, use a neutral prior so R_MS isn't stuck
        if opponent_value == 0:
            opponent_value = MAX_GAME_VALUE / 2.0  # neutral assumption

        # attack_count useful for plotting/debugging
        attack_count = sum(1 for vr in self.units(UnitTypeId.VOIDRAY) if vr.is_attacking and vr.target_in_range)

        # include info dict fields expected by trainer
        info = {
            "self_value": float(self_value),
            "opponent_value": float(opponent_value),
            "heuristic_reward": float(reward),
            "attack_count": int(attack_count),
            "units_count": int(self.units.amount)
        }

        # finally push step result
        self.result_out.put({
            "observation": obs,
            "reward": float(reward),
            "action": None,
            "done": False,
            "truncated": False,
            "info": info
        })

        
#        self_value = self.minerals + self.vespene + 10 * self.units(UnitTypeId.VOIDRAY).amount
        
        
#        opponent_value = 0
        #if iteration % 10 == 0:
        #    print(f"Reward: {reward}")

#        self.result_out.put({"observation" : obs, "reward" : reward, "action" : None, "done" : False, "truncated" : False, "info" : {"self_value":self_value, "heuristic_reward":reward}})

    async def build_workers(self):
        for nexus in self.structures(UnitTypeId.NEXUS).ready:
            if nexus.is_idle and self.can_afford(UnitTypeId.PROBE):
                nexus.train(UnitTypeId.PROBE)

    async def build_pylons(self):
        if not self.already_pending(UnitTypeId.PYLON):
            if self.supply_left < 5:
                if len(self.structures(UnitTypeId.NEXUS)) > 0:
                    nexus = self.structures(UnitTypeId.NEXUS).ready.random
                    position = nexus.position.towards(self.game_info.map_center, 5)
                    if self.can_afford(UnitTypeId.PYLON):
                        await self.build(UnitTypeId.PYLON, near=position)

    async def build_assimilators(self):
        if len(self.structures(UnitTypeId.NEXUS)) > 0:
            for nexus in self.structures(UnitTypeId.NEXUS).ready:
                vespenes = self.vespene_geyser.closer_than(15.0, nexus.position)  # Corrected to use `position`
                for vespene in vespenes:
                    if not self.can_afford(UnitTypeId.ASSIMILATOR) or self.structures(UnitTypeId.ASSIMILATOR).closer_than(1.0, vespene).exists:
                        continue
                    worker = self.select_build_worker(vespene.position)
                    if worker:
                        worker.build(UnitTypeId.ASSIMILATOR, vespene)

    async def expand(self):
        if self.can_afford(UnitTypeId.NEXUS) and not self.already_pending(UnitTypeId.NEXUS):
            location = await self.get_next_expansion()  # Assuming this is async
            if location:
                await self.build(UnitTypeId.NEXUS, near=location)

    async def build_gateway(self):
        if len(self.structures(UnitTypeId.NEXUS)) > 0:
            if not self.structures(UnitTypeId.GATEWAY).ready.exists and not self.already_pending(UnitTypeId.GATEWAY):
                pylons = self.structures(UnitTypeId.PYLON).ready
                if pylons.exists and self.can_afford(UnitTypeId.GATEWAY):
                    await self.build(UnitTypeId.GATEWAY, near=pylons.closest_to(self.structures(UnitTypeId.NEXUS).first))

    async def build_cybernetics_core(self):
        if len(self.structures(UnitTypeId.NEXUS)) > 0:
            if self.structures(UnitTypeId.GATEWAY).ready.exists and not self.structures(UnitTypeId.CYBERNETICSCORE):
                pylon = self.structures(UnitTypeId.PYLON).ready.random
                if self.can_afford(UnitTypeId.CYBERNETICSCORE) and not self.already_pending(UnitTypeId.CYBERNETICSCORE):
                    await self.build(UnitTypeId.CYBERNETICSCORE, near=pylon.position.towards(self.game_info.map_center, 5))

    async def build_stargates(self):
        if len(self.structures(UnitTypeId.NEXUS)) > 0:
            if self.structures(UnitTypeId.CYBERNETICSCORE).ready.exists:
                for pylon in self.structures(UnitTypeId.PYLON).ready:
                    if self.can_afford(UnitTypeId.STARGATE) and not self.already_pending(UnitTypeId.STARGATE):
                        await self.build(UnitTypeId.STARGATE, near=pylon.position.towards(self.game_info.map_center, 5))

    async def build_void_rays(self):
        for stargate in self.structures(UnitTypeId.STARGATE).ready.idle:
            if self.can_afford(UnitTypeId.VOIDRAY) and self.supply_left > 0:
                stargate.train(UnitTypeId.VOIDRAY)

    async def defend_bases(self):
        if self.units(UnitTypeId.VOIDRAY).amount < 5:
            for vr in self.units(UnitTypeId.VOIDRAY).idle:
                vr.move(self.start_location)

    async def attack(self):
        # take all void rays and attack!
        for voidray in self.units(UnitTypeId.VOIDRAY).idle:
            if self.enemy_units.closer_than(10, voidray):
                voidray.attack(random.choice(self.enemy_units.closer_than(10, voidray)))
            elif self.enemy_units:
                voidray.attack(random.choice(self.enemy_units))
            elif self.enemy_structures.closer_than(10, voidray):
                voidray.attack(random.choice(self.enemy_structures.closer_than(10, voidray)))
            elif self.enemy_structures:
                voidray.attack(random.choice(self.enemy_structures))
            elif self.enemy_start_locations:
                voidray.attack(self.enemy_start_locations[0])
        
    def reward_function(self):
        # micro reward for valid attacks (keeps previous behaviour)
        reward = 0.0
        attack_count = 0
        for vr in self.units(UnitTypeId.VOIDRAY):
            if vr.is_attacking and vr.target_in_range:
                if self.enemy_units.closer_than(12, vr) or self.enemy_structures.closer_than(12, vr):
                    reward += 0.05  # reduced per-attack reward
                    attack_count += 1

        # small economy/army size bonus to encourage balanced macro (dense)
        economy_bonus = (self.minerals / 1000.0) + (self.vespene / 1000.0)
        army_bonus = 0.02 * self.units.amount  # small per-unit weight

        reward += 0.1 * economy_bonus + 0.01 * army_bonus

        # small inactivity penalty if no attacks for long time (prevents too-safe stagnation)
        if attack_count == 0 and self.time > 60.0:
            reward -= 0.01  # tiny penalty over time

        return float(reward)
    
    def compute_enemy_state(self):
        # Count enemy units we can currently see
        enemy_voidrays = self.enemy_units(UnitTypeId.VOIDRAY).amount
        enemy_army_units = self.enemy_units.not_structure.amount
        enemy_structures = self.enemy_structures.amount
    
        # If minerals/vespene cannot be observed, keep last seen
        if self.state.observation.player.minerals is not None:
            self.enemy_minerals_seen = self.state.observation.player.minerals
        if self.state.observation.player.vespene is not None:
            self.enemy_vespene_seen = self.state.observation.player.vespene
    
        self.enemy_voidrays = enemy_voidrays
        self.enemy_army_units = enemy_army_units
        self.enemy_structures = enemy_structures



    
    def visualize_intel(self):

        obs = np.zeros((224, 224, 3), dtype=np.uint8)
        """ limits = np.array(constants.OBSERVATION_SPACE_ARRAY)

        # Number of probes
        obs[0] = self.units(UnitTypeId.PROBE).amount
        # Number of void rays
        obs[1] = self.units(UnitTypeId.VOIDRAY).amount
        # Number of attacking void rays
        obs[2] = len([vr for vr in self.units(UnitTypeId.VOIDRAY) if vr.is_attacking])
        # Number of nexus
        obs[3] = self.units(UnitTypeId.NEXUS).amount
        # Number of assimilators
        obs[4] = self.units(UnitTypeId.ASSIMILATOR).amount
        # Number of stargates
        obs[5] = self.units(UnitTypeId.STARGATE).amount
        # Number of pylons
        obs[6] = self.units(UnitTypeId.PYLON).amount
        # Supply left
        obs[7] = min(199, int(self.supply_left))
        # Game time
        obs[8] = self.time

        for i in range(len(obs)):
            if obs[i] >= limits[i]-1:
                obs[i] = limits[i] - 1 """

        # Draw our units
        for our_unit in self.all_own_units:
            # if it is a voidray:
            if our_unit.type_id == UnitTypeId.VOIDRAY:
                pos = our_unit.position
                c = [255, 75 , 75]
                # get health:
                fraction = our_unit.health / our_unit.health_max if our_unit.health_max > 0 else 0.0001
                obs[math.ceil(pos.y)][math.ceil(pos.x)] = [int(fraction*i) for i in c]


            else:
                pos = our_unit.position
                c = [175, 255, 0]
                # get health:
                fraction = our_unit.health / our_unit.health_max if our_unit.health_max > 0 else 0.0001
                obs[math.ceil(pos.y)][math.ceil(pos.x)] = [int(fraction*i) for i in c]

        # Draw the minerals
        for mineral in self.mineral_field:
            pos = mineral.position
            c = [175, 255, 255]
            fraction = mineral.mineral_contents / 1800
            if mineral.is_visible:
                obs[math.ceil(pos.y)][math.ceil(pos.x)] = c
            else:
                obs[math.ceil(pos.y)][math.ceil(pos.x)] = [20,75,50]  

        # Draw the enemy start location
        for pos in self.enemy_start_locations:
            c = [0, 0, 255]
            obs[math.ceil(pos.y)][math.ceil(pos.x)] = c

        # Draw the enemy units
        for enemy_unit in self.all_enemy_units:
            pos = enemy_unit.position
            c = [100, 0, 255]
            # get unit health fraction:
            fraction = enemy_unit.health / enemy_unit.health_max if enemy_unit.health_max > 0 else 0.0001
            obs[math.ceil(pos.y)][math.ceil(pos.x)] = [int(fraction*i) for i in c]

        # Draw the enemy structures
        for enemy_structure in self.enemy_structures:
            pos = enemy_structure.position
            c = [0, 100, 255]
            # get structure health fraction:
            fraction = enemy_structure.health / enemy_structure.health_max if enemy_structure.health_max > 0 else 0.0001
            obs[math.ceil(pos.y)][math.ceil(pos.x)] = [int(fraction*i) for i in c]

        # Draw our structures
        for our_structure in self.all_own_units.structure:
            if our_structure.type_id == UnitTypeId.NEXUS:
                pos = our_structure.position
                c = [255, 255, 175]
                # get structure health fraction:
                fraction = our_structure.health / our_structure.health_max if our_structure.health_max > 0 else 0.0001
                obs[math.ceil(pos.y)][math.ceil(pos.x)] = [int(fraction*i) for i in c]
            
            else:
                pos = our_structure.position
                c = [0, 255, 175]
                # get structure health fraction:
                fraction = our_structure.health / our_structure.health_max if our_structure.health_max > 0 else 0.0001
                obs[math.ceil(pos.y)][math.ceil(pos.x)] = [int(fraction*i) for i in c]

        # Draw the vespene geysers
        for vespene in self.vespene_geyser:
            pos = vespene.position
            c = [255, 175, 255]
            fraction = vespene.vespene_contents / 2250

            if vespene.is_visible:
                obs[math.ceil(pos.y)][math.ceil(pos.x)] = [int(fraction*i) for i in c]
            else:
                obs[math.ceil(pos.y)][math.ceil(pos.x)] = [50,20,75]



        # Show obs with OpenCV, resized to be larger
        cv2.imshow('obs',cv2.flip(cv2.resize(obs, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST), 0))
        cv2.waitKey(1)
        
        return obs
