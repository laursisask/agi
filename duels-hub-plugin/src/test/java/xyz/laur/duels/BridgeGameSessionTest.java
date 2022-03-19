package xyz.laur.duels;

import com.google.common.base.Function;
import com.google.common.base.Functions;
import org.bukkit.Location;
import org.bukkit.Material;
import org.bukkit.Server;
import org.bukkit.World;
import org.bukkit.block.Block;
import org.bukkit.entity.*;
import org.bukkit.event.block.BlockBreakEvent;
import org.bukkit.event.block.BlockPlaceEvent;
import org.bukkit.event.entity.EntityDamageByEntityEvent;
import org.bukkit.event.entity.EntityDamageEvent;
import org.bukkit.event.entity.EntityShootBowEvent;
import org.bukkit.event.player.PlayerMoveEvent;
import org.bukkit.event.player.PlayerQuitEvent;
import org.bukkit.inventory.ItemStack;
import org.bukkit.inventory.PlayerInventory;
import org.bukkit.plugin.Plugin;
import org.bukkit.plugin.PluginManager;
import org.bukkit.scheduler.BukkitScheduler;
import org.junit.Before;
import org.junit.Test;
import org.mockito.ArgumentCaptor;

import java.util.HashMap;
import java.util.Map;
import java.util.logging.Logger;

import static org.bukkit.event.entity.EntityDamageEvent.DamageCause.ENTITY_ATTACK;
import static org.bukkit.event.entity.EntityDamageEvent.DamageCause.FALL;
import static org.junit.Assert.*;
import static org.mockito.Matchers.any;
import static org.mockito.Mockito.*;
import static xyz.laur.duels.BridgeGameSession.MAX_DURATION;
import static xyz.laur.duels.BridgeGameSession.Team.BLUE;
import static xyz.laur.duels.BridgeMap.TREEHOUSE;
import static xyz.laur.duels.TestHelper.mockServer;

public class BridgeGameSessionTest {
    private SessionManager sessionManager;
    private InvisibilityManager invisibilityManager;
    private BridgeGameSession session;
    private World world;
    private BukkitScheduler scheduler;
    private SkinChanger skinChanger;
    private Plugin plugin;

    @Before
    public void setUp() {
        mockServer();

        sessionManager = new SessionManager();
        invisibilityManager = mock(InvisibilityManager.class);
        this.plugin = mock(Plugin.class);
        when(plugin.getLogger()).thenReturn(Logger.getGlobal());

        Server server = mock(Server.class);
        PluginManager pluginManager = mock(PluginManager.class);
        scheduler = mock(BukkitScheduler.class);
        when(server.getPluginManager()).thenReturn(pluginManager);
        when(server.getScheduler()).thenReturn(scheduler);
        when(plugin.getServer()).thenReturn(server);

        world = mock(World.class);
        Block block = mock(Block.class);
        when(block.getType()).thenReturn(Material.BARRIER);
        when(block.getRelative(any())).thenReturn(block);

        when(world.getBlockAt(any(Location.class))).thenReturn(block);

        skinChanger = mock(SkinChanger.class);

        session = new BridgeGameSession(world, sessionManager, invisibilityManager, skinChanger, plugin,
                TREEHOUSE);
    }

    @Test
    public void testGameMainLifecycle() {
        Player player1 = createMockPlayer();
        session.addPlayer(player1);

        verify(invisibilityManager, times(1)).update();

        assertEquals(GameState.WAITING_FOR_PLAYERS, session.getState());
        assertTrue(session.hasPlayer(player1));
        verify(player1, never()).teleport(any(Location.class));

        Player player2 = createMockPlayer();
        session.addPlayer(player2);

        assertEquals(GameState.PLAYING, session.getState());
        assertTrue(session.hasPlayer(player1));
        assertTrue(session.hasPlayer(player2));
        verify(player1, times(1)).sendMessage("Game started");
        verify(player2, times(1)).sendMessage("Game started");

        verify(invisibilityManager, times(2)).update();

        verify(player1, times(1)).teleport(any(Location.class));
        verify(player2, times(1)).teleport(any(Location.class));

        verify(player1.getInventory(), times(1)).clear();
        verify(player2.getInventory(), times(1)).clear();

        // Random entity dying
        Sheep sheep = mock(Sheep.class);
        session.onEntityDamage(createDamageEvent(sheep, 10));
        assertEquals(GameState.PLAYING, session.getState());

        // Player in another game dying
        Player unrelatedPlayer = mock(Player.class);
        session.onEntityDamage(createDamageEvent(unrelatedPlayer, 10));
        assertEquals(GameState.PLAYING, session.getState());

        // Player in the same game taking damage but not dying
        when(player2.getHealth()).thenReturn(7D);
        EntityDamageEvent event1 = createDamageEvent(player2, 5);
        session.onEntityDamage(event1);
        assertFalse(event1.isCancelled());
        assertEquals(GameState.PLAYING, session.getState());
        // Check that player is not respawned (teleported) after taking damage
        verify(player2, times(1)).teleport(any(Location.class));

        // Player in the same game taking fall damage
        EntityDamageEvent event2 = createDamageEvent(player2, 4, FALL);
        session.onEntityDamage(event2);
        assertTrue(event2.isCancelled());
        assertEquals(GameState.PLAYING, session.getState());
        // Check that player is not respawned (teleported) after taking damage
        verify(player2, times(1)).teleport(any(Location.class));

        // Player in the same game dying
        when(player1.getHealth()).thenReturn(8D);
        EntityDamageEvent event3 = createDamageEvent(player1, 11);
        session.onEntityDamage(event3);
        assertTrue(event3.isCancelled());
        assertEquals(GameState.PLAYING, session.getState());
        verify(player1, times(2)).teleport(any(Location.class));
        verify(player2, times(1)).teleport(any(Location.class));

        // Player falling to void
        verify(player1, times(0)).sendMessage(startsWith("metadata:fell_to_void"));
        Location from = new Location(world, 137, 86, 54);
        Location to = new Location(world, 132, 76, 33);
        session.onMove(new PlayerMoveEvent(player1, from, to));
        assertEquals(GameState.PLAYING, session.getState());
        verify(player1, times(3)).teleport(any(Location.class));
        verify(player2, times(1)).teleport(any(Location.class));
        verify(player1, times(1)).sendMessage("metadata:fell_to_void:1.00000");

        // Player jumping to their own hole
        verify(player1, times(0)).sendMessage(startsWith("metadata:fell_to_own_hole"));

        Location ownHole = session.getPlayerTeam(player1) == BLUE ?
                TREEHOUSE.getBlueHole().clone() : TREEHOUSE.getRedHole().clone();
        ownHole.setWorld(world);
        ownHole.add(1, 1, 3);

        session.onMove(new PlayerMoveEvent(player1, from, ownHole));
        assertEquals(GameState.PLAYING, session.getState());
        verify(player1, times(4)).teleport(any(Location.class));
        verify(player2, times(1)).teleport(any(Location.class));
        verify(player1, times(1)).sendMessage("metadata:fell_to_own_hole:1.00000");

        verify(player1, times(0)).sendMessage("metadata:round_won:1.00000");
        verify(player1, times(0)).sendMessage("metadata:round_lost:1.00000");
        verify(player2, times(0)).sendMessage("metadata:round_won:1.00000");
        verify(player2, times(0)).sendMessage("metadata:round_lost:1.00000");

        // 1st player jumping to other player's hole
        Location otherHole = session.getPlayerTeam(player1) == BLUE ?
                TREEHOUSE.getRedHole().clone() : TREEHOUSE.getBlueHole().clone();
        otherHole.setWorld(world);
        otherHole.add(1, 1, 3);
        session.onMove(new PlayerMoveEvent(player1, from, otherHole));

        verify(player1, times(5)).teleport(any(Location.class));
        verify(player2, times(2)).teleport(any(Location.class));

        verify(player1, times(1)).sendMessage("metadata:round_won:1.00000");
        verify(player1, times(0)).sendMessage("metadata:round_lost:1.00000");

        verify(player2, times(0)).sendMessage("metadata:round_won:1.00000");
        verify(player2, times(1)).sendMessage("metadata:round_lost:1.00000");

        // 1st player jumps to other player's hole 2 times more
        session.onMove(new PlayerMoveEvent(player1, from, otherHole));
        session.onMove(new PlayerMoveEvent(player1, from, otherHole));
        assertEquals(GameState.PLAYING, session.getState());

        // 1st player jumps to other player's hole one more time and wins the game
        session.onMove(new PlayerMoveEvent(player1, from, otherHole));
        assertEquals(GameState.ENDED, session.getState());

        verify(player1, times(1)).sendMessage("You won");
        verify(player1, times(1)).sendMessage("metadata:win:1.00000");

        verify(player2, times(1)).sendMessage("You lost");
        verify(player2, never()).sendMessage(startsWith("metadata:win"));
    }

    private EntityDamageEvent createDamageEvent(Entity damagee, double damage) {
        return createDamageEvent(damagee, damage, ENTITY_ATTACK);
    }

    private EntityDamageEvent createDamageEvent(Entity damagee, double damage, EntityDamageEvent.DamageCause cause) {
        Map<EntityDamageEvent.DamageModifier, Double> modifiers = new HashMap<>();
        modifiers.put(EntityDamageEvent.DamageModifier.BASE, damage);
        Map<EntityDamageEvent.DamageModifier, Function<? super Double, Double>> modifierFunctions = new HashMap<>();
        modifierFunctions.put(EntityDamageEvent.DamageModifier.BASE, Functions.constant(0D));

        return new EntityDamageEvent(damagee, cause, modifiers,
                modifierFunctions);
    }

    @Test
    public void testPlayerQuits() {
        Player player1 = createMockPlayer();
        session.addPlayer(player1);

        Player player2 = createMockPlayer();
        session.addPlayer(player2);

        assertEquals(GameState.PLAYING, session.getState());

        session.onQuit(new PlayerQuitEvent(player2, null));
        assertEquals(GameState.ENDED, session.getState());

        verify(player1, times(1)).sendMessage("You won");
        verify(player1, times(1)).sendMessage("metadata:win:1.00000");

        verify(player2, times(1)).sendMessage("You lost");
        verify(player2, never()).sendMessage(startsWith("metadata:win"));
    }

    @Test
    public void testTimeout() {
        Player player1 = createMockPlayer();
        session.addPlayer(player1);

        Player player2 = createMockPlayer();
        session.addPlayer(player2);

        assertEquals(GameState.PLAYING, session.getState());

        ArgumentCaptor<Runnable> task = ArgumentCaptor.forClass(Runnable.class);
        verify(scheduler, times(1))
                .scheduleSyncDelayedTask(any(Plugin.class), task.capture(), eq(MAX_DURATION));

        task.getValue().run();
        assertEquals(GameState.ENDED, session.getState());
        verify(player1, times(1)).sendMessage("You lost");
        verify(player1, never()).sendMessage(startsWith("metadata:win"));

        verify(player2, times(1)).sendMessage("You lost");
        verify(player2, never()).sendMessage(startsWith("metadata:win"));
    }

    @Test
    public void testDirectAttack() {
        Player player1 = createMockPlayer();
        session.addPlayer(player1);

        Player player2 = createMockPlayer();
        session.addPlayer(player2);

        EntityDamageByEntityEvent damageEvent = mock(EntityDamageByEntityEvent.class);
        when(damageEvent.getFinalDamage()).thenReturn(3.5D);
        when(damageEvent.getDamager()).thenReturn(player1);
        when(damageEvent.getEntity()).thenReturn(player2);

        session.onPlayerDamage(damageEvent);

        verify(player1, times(1)).sendMessage("metadata:hits_done:3.50000");
        verify(player2, times(1)).sendMessage("metadata:hits_received:3.50000");
    }

    @Test
    public void testBowAttack() {
        Player player1 = createMockPlayer();
        session.addPlayer(player1);

        Player player2 = createMockPlayer();
        session.addPlayer(player2);

        EntityDamageByEntityEvent damageEvent = mock(EntityDamageByEntityEvent.class);
        when(damageEvent.getFinalDamage()).thenReturn(3.5D);
        when(damageEvent.getEntity()).thenReturn(player2);

        Arrow arrow = mock(Arrow.class);
        when(arrow.getShooter()).thenReturn(player1);
        when(damageEvent.getDamager()).thenReturn(arrow);

        session.onPlayerDamage(damageEvent);

        verify(damageEvent, never()).setCancelled(anyBoolean());

        verify(player1, times(1)).sendMessage("metadata:hits_done:3.50000");
        verify(player2, times(1)).sendMessage("metadata:hits_received:3.50000");
    }

    @Test
    public void testBowAttackByUnrelatedArrow() {
        Player player1 = createMockPlayer();
        session.addPlayer(player1);

        Player player2 = createMockPlayer();
        session.addPlayer(player2);

        EntityDamageByEntityEvent damageEvent = mock(EntityDamageByEntityEvent.class);
        when(damageEvent.getFinalDamage()).thenReturn(3.5D);
        when(damageEvent.getEntity()).thenReturn(player2);

        Arrow arrow = mock(Arrow.class);
        when(arrow.getType()).thenReturn(EntityType.ARROW);
        when(arrow.getShooter()).thenReturn(mock(Player.class));
        when(damageEvent.getDamager()).thenReturn(arrow);

        session.onPlayerDamage(damageEvent);

        verify(damageEvent, times(1)).setCancelled(eq(true));

        verify(player1, never()).sendMessage("metadata:hits_done:3.50000");
        verify(player2, never()).sendMessage("metadata:hits_received:3.50000");
    }

    @Test
    public void testShootArrowMetadata() {
        Player player1 = createMockPlayer();
        session.addPlayer(player1);

        Player player2 = createMockPlayer();
        session.addPlayer(player2);

        Arrow arrow = mock(Arrow.class);
        EntityShootBowEvent event = new EntityShootBowEvent(player1, new ItemStack(Material.BOW), arrow, 0.5F);
        session.onBowShoot(event);

        verify(player1, times(1)).sendMessage("metadata:shoot_arrow:0.50000");
        verify(player2, never()).sendMessage("metadata:shoot_arrow:0.50000");
    }

    @Test
    public void testShootArrowUnrelatedPlayer() {
        Player player1 = createMockPlayer();
        session.addPlayer(player1);

        Player player2 = createMockPlayer();
        session.addPlayer(player2);

        Player player3 = createMockPlayer();

        Arrow arrow = mock(Arrow.class);
        EntityShootBowEvent event = new EntityShootBowEvent(player3, new ItemStack(Material.BOW), arrow, 0.5F);
        session.onBowShoot(event);

        verify(player3, never()).sendMessage(anyString());
    }

    @Test
    public void testBreakBlock() {
        Player player1 = createMockPlayer();
        session.addPlayer(player1);

        Player player2 = createMockPlayer();
        session.addPlayer(player2);

        Block grass = mock(Block.class);
        when(grass.getType()).thenReturn(Material.GRASS);

        BlockBreakEvent event1 = new BlockBreakEvent(grass, player1);
        session.onBlockBreak(event1);
        assertTrue(event1.isCancelled());

        Block clay = mock(Block.class);
        when(clay.getType()).thenReturn(Material.STAINED_CLAY);

        BlockBreakEvent event2 = new BlockBreakEvent(clay, player1);
        session.onBlockBreak(event2);
        assertFalse(event2.isCancelled());
    }

    @Test
    public void testPlaceBlock() {
        Player player1 = createMockPlayer();
        session.addPlayer(player1);

        Player player2 = createMockPlayer();
        session.addPlayer(player2);

        Block block1 = mock(Block.class);
        when(block1.getLocation()).thenReturn(new Location(world, 3, 64, 7));

        BlockPlaceEvent event1 = new BlockPlaceEvent(block1, null, null, null, player1, true);
        session.onBlockPlace(event1);
        assertFalse(event1.isCancelled());

        Block block2 = mock(Block.class);
        Location nearHole = TREEHOUSE.getBlueHole().clone();
        nearHole.setWorld(world);
        nearHole.add(1, 1, 2);

        when(block2.getLocation()).thenReturn(nearHole);

        BlockPlaceEvent event2 = new BlockPlaceEvent(block2, null, null, null, player1, true);
        session.onBlockPlace(event2);
        assertTrue(event2.isCancelled());
    }

    @Test
    public void testMovementMetadata() {
        Player player1 = createMockPlayer();
        when(player1.getLocation()).thenReturn(new Location(world, 10, 20, 30));
        session.addPlayer(player1);

        Player player2 = createMockPlayer();
        when(player2.getLocation()).thenReturn(new Location(world, 5, 80, 15));
        session.addPlayer(player2);

        ArgumentCaptor<Runnable> task = ArgumentCaptor.forClass(Runnable.class);
        verify(scheduler, times(1)).scheduleSyncRepeatingTask(any(), task.capture(), eq(0L), eq(10L));

        // Without change to location there should be no metadata sent
        for (int i = 0; i < 20; i++) {
            task.getValue().run();
        }

        verify(player1, never()).sendMessage(contains("distance_change"));
        verify(player2, never()).sendMessage(contains("distance_change"));

        // Player 2 moves closer to the hole
        when(player2.getLocation()).thenReturn(new Location(world, 7, 85, 10));
        task.getValue().run();

        verify(player1, never()).sendMessage(contains("distance_change"));
        verify(player2, times(1)).sendMessage(contains("distance_change"));

        if (session.getPlayerTeam(player2) == BLUE) {
            // Red hole is at (33, 89, 0)
            // Move from (5, 80, 15) to (7, 85, 10)
            // Distance before: 33.015148038438355
            // Distance after: 28.142494558940577
            // Change: -4.872653479497778
            verify(player2, times(1)).sendMessage("metadata:distance_change:-4.87265");
        } else {
            // Blue hole is at (-33, 89, 0)
            // Move from (5, 80, 15) to (7, 85, 10)
            // Distance before 41.83300132670378
            // Distance after: 41.42463035441596
            // Change: -0.4083709722878197
            verify(player2, times(1)).sendMessage("metadata:distance_change:-0.40837");
        }
    }

    protected Player createMockPlayer() {
        Player player = mock(Player.class);
        PlayerInventory inventory = mock(PlayerInventory.class);
        when(player.getInventory()).thenReturn(inventory);

        return player;
    }
}
