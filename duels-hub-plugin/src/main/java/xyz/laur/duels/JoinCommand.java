package xyz.laur.duels;

import org.bukkit.ChatColor;
import org.bukkit.World;
import org.bukkit.command.Command;
import org.bukkit.command.CommandExecutor;
import org.bukkit.command.CommandSender;
import org.bukkit.entity.Player;
import org.bukkit.plugin.Plugin;

public class JoinCommand implements CommandExecutor {
    private final SessionManager sessionManager;
    private final InvisibilityManager invisibilityManager;
    private final Plugin plugin;
    private final World world;

    public JoinCommand(SessionManager sessionManager, InvisibilityManager invisibilityManager, Plugin plugin, World world) {
        this.sessionManager = sessionManager;
        this.invisibilityManager = invisibilityManager;
        this.plugin = plugin;
        this.world = world;
    }

    @Override
    public boolean onCommand(CommandSender sender, Command command, String label, String[] args) {
        if (!(sender instanceof Player)) {
            sender.sendMessage(ChatColor.RED + "Only players can join games");
            return true;
        }

        if (args.length != 3) {
            return false;
        }

        String sessionName = args[0];
        float randomizationFactor = Float.parseFloat(args[1]);

        GameSession.GameMap map = GameSession.GameMap.fromName(args[2]);

        GameSession session = sessionManager.getByName(sessionName);

        if (session == null) {
            plugin.getLogger().info("Creating new session");
            session = new GameSession(world, sessionManager, invisibilityManager, plugin, randomizationFactor, map);
            sessionManager.addSession(sessionName, session);
        }

        session.addPlayer((Player) sender);

        return true;
    }
}
