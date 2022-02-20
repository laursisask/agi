package xyz.laur.duels;

import org.bukkit.ChatColor;
import org.bukkit.Location;
import org.bukkit.World;
import org.bukkit.WorldCreator;
import org.bukkit.craftbukkit.v1_8_R3.entity.CraftPlayer;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.event.block.BlockBreakEvent;
import org.bukkit.event.entity.EntityDamageEvent;
import org.bukkit.event.entity.FoodLevelChangeEvent;
import org.bukkit.event.entity.ProjectileHitEvent;
import org.bukkit.event.player.PlayerJoinEvent;
import org.bukkit.event.weather.WeatherChangeEvent;
import org.bukkit.event.world.ChunkUnloadEvent;
import org.bukkit.plugin.java.JavaPlugin;
import org.bukkit.scoreboard.*;

import java.util.Arrays;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

public class DuelsHubPlugin extends JavaPlugin implements Listener {
    private SessionManager sessionManager;
    private InvisibilityManager invisibilityManager;
    private EmblemRenderer emblemRenderer;
    private Map<ClassicGameSession.GameMap, MapBarrier> barriers;

    @Override
    public void onEnable() {
        World sumoWorld = new WorldCreator("sumo").createWorld();
        sumoWorld.setThundering(false);
        sumoWorld.setStorm(false);
        sumoWorld.setSpawnFlags(false, false);

        World classicWorld = new WorldCreator("classic").createWorld();
        classicWorld.setThundering(false);
        classicWorld.setStorm(false);
        classicWorld.setSpawnFlags(false, false);
        preloadClassicChunks(classicWorld);

        sessionManager = new SessionManager();
        invisibilityManager = new InvisibilityManager(sessionManager, getServer());
        invisibilityManager.update();

        SkinChanger skinChanger = new SkinChanger(this);

        barriers = Arrays.stream(ClassicGameSession.GameMap.values())
                .collect(Collectors.toMap(Function.identity(), map -> new MapBarrier(this, map, classicWorld)));

        getCommand("sumo").setExecutor(new SumoJoinCommand(sessionManager, invisibilityManager, skinChanger,
                this, sumoWorld));
        getCommand("classic").setExecutor(new ClassicJoinCommand(sessionManager, invisibilityManager, skinChanger, barriers,
                this, classicWorld));
        getCommand("games").setExecutor(new GamesCommand(sessionManager));

        getServer().getPluginManager().registerEvents(this, this);

        showPlayerHealth();
        createTeam();

        emblemRenderer = new EmblemRenderer(this, sumoWorld, classicWorld);
        emblemRenderer.start();

        getServer().getScheduler().scheduleSyncRepeatingTask(this, this::removeStuckArrows, 0, 200);
    }

    @Override
    public void onDisable() {
        if (emblemRenderer != null) {
            emblemRenderer.stop();
        }

        if (barriers != null) {
            barriers.values().forEach(MapBarrier::removeBarrier);
        }
    }

    @EventHandler
    public void onChunkUnload(ChunkUnloadEvent event) {
        event.setCancelled(true);
    }

    @EventHandler
    public void onFoodLevelChange(FoodLevelChangeEvent event) {
        event.setCancelled(true);
    }

    @EventHandler
    public void onDamage(EntityDamageEvent event) {
        if (!(event.getEntity() instanceof Player)) {
            return;
        }

        Player player = (Player) event.getEntity();
        GameSession session = sessionManager.getPlayerSession(player);

        if (session == null || session.getState() != GameState.PLAYING) {
            event.setDamage(0);
        }
    }

    @EventHandler
    public void onBreakBlock(BlockBreakEvent event) {
        if (!event.getPlayer().isOp()) {
            event.setCancelled(true);
        }
    }

    @EventHandler
    public void onProjectileHit(ProjectileHitEvent event) {
        event.getEntity().remove();
    }

    @EventHandler
    public void onJoin(PlayerJoinEvent event) {
        invisibilityManager.update();

        Team team = getServer().getScoreboardManager().getMainScoreboard().getTeam("all");
        team.addEntry(event.getPlayer().getName());
    }

    @EventHandler
    public void onWeatherChange(WeatherChangeEvent event) {
        if (event.toWeatherState()) {
            event.setCancelled(true);
        }
    }

    private void removeStuckArrows() {
        for (Player player : getServer().getOnlinePlayers()) {
            ((CraftPlayer) player).getHandle().getDataWatcher().watch(9, (byte) 0);
        }
    }

    private void showPlayerHealth() {
        ScoreboardManager sm = getServer().getScoreboardManager();
        Scoreboard scoreboard = sm.getMainScoreboard();

        Objective existingObjective = scoreboard.getObjective("showhealth");
        if (existingObjective != null) {
            existingObjective.unregister();
        }

        Objective objective = scoreboard.registerNewObjective("showhealth", Criterias.HEALTH);
        objective.setDisplaySlot(DisplaySlot.BELOW_NAME);
        objective.setDisplayName(ChatColor.RED + "❤");
    }

    private void createTeam() {
        ScoreboardManager sm = getServer().getScoreboardManager();
        Scoreboard scoreboard = sm.getMainScoreboard();

        Team existingTeam = scoreboard.getTeam("all");
        if (existingTeam != null) {
            existingTeam.unregister();
        }

        Team team = scoreboard.registerNewTeam("all");
        team.setPrefix(ChatColor.RED.toString());
    }

    // When a chunk is not loaded and player is teleported to it, then the player will
    // get stuck inside a block and die. To prevent that from happening, preload chunks
    // where players are teleported when the game starts.
    private void preloadClassicChunks(World world) {
        for (ClassicGameSession.GameMap map : ClassicGameSession.GameMap.values()) {
            Location spawnLoc1 = map.getSpawnLocation1().clone();
            spawnLoc1.setWorld(world);
            if (!spawnLoc1.getChunk().load()) {
                throw new RuntimeException("Failed to preload chunk");
            }

            Location spawnLoc2 = map.getSpawnLocation2().clone();
            spawnLoc2.setWorld(world);
            if (!spawnLoc2.getChunk().load()) {
                throw new RuntimeException("Failed to preload chunk");
            }
        }
    }
}
