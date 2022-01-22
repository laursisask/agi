package xyz.laur.duels;

import org.bukkit.ChatColor;
import org.bukkit.World;
import org.bukkit.command.Command;
import org.bukkit.command.CommandExecutor;
import org.bukkit.command.CommandSender;
import org.bukkit.entity.Player;
import org.bukkit.plugin.Plugin;

public class SumoJoinCommand implements CommandExecutor {
    private final SessionManager sessionManager;
    private final InvisibilityManager invisibilityManager;
    private final SkinChanger skinChanger;
    private final Plugin plugin;
    private final World world;

    public SumoJoinCommand(SessionManager sessionManager, InvisibilityManager invisibilityManager,
                           SkinChanger skinChanger, Plugin plugin, World world) {
        this.sessionManager = sessionManager;
        this.invisibilityManager = invisibilityManager;
        this.skinChanger = skinChanger;
        this.plugin = plugin;
        this.world = world;
    }

    @Override
    public boolean onCommand(CommandSender sender, Command command, String label, String[] args) {
        if (!(sender instanceof Player)) {
            sender.sendMessage(ChatColor.RED + "Only players can join games");
            return true;
        }

        if (args.length != 4) {
            return false;
        }

        String sessionName = args[0];
        float randomizationFactor = Float.parseFloat(args[1]);
        SumoGameSession.GameMap map = SumoGameSession.GameMap.fromName(args[2]);
        boolean randomTeleport = Boolean.parseBoolean(args[3]);

        SumoGameSession session = (SumoGameSession) sessionManager.getByName(sessionName);

        if (session == null) {
            plugin.getLogger().info("Creating new session");
            session = new SumoGameSession(world, sessionManager, invisibilityManager, skinChanger, plugin,
                    randomizationFactor, map, randomTeleport);
            sessionManager.addSession(sessionName, session);
        }

        session.addPlayer((Player) sender);

        return true;
    }
}
