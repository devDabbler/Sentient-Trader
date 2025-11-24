"""
Windows Services Manager for Sentient Trader

Centralized management script for all Sentient Trader Windows services:
1. Stock Informational Monitor
2. DEX Launch Monitor
3. Crypto Breakout Monitor

Usage:
    python windows_services\manage_services.py [command] [service]

Commands:
    install-all     - Install all services
    uninstall-all   - Uninstall all services
    start-all       - Start all services
    stop-all        - Stop all services
    status          - Show status of all services
    
    install [name]  - Install specific service
    uninstall [name]- Uninstall specific service
    start [name]    - Start specific service
    stop [name]     - Stop specific service
    restart [name]  - Restart specific service

Services:
    stock           - Stock Informational Monitor
    dex             - DEX Launch Monitor
    crypto          - Crypto Breakout Monitor

Examples:
    python windows_services\manage_services.py install-all
    python windows_services\manage_services.py start stock
    python windows_services\manage_services.py status
"""

import sys
import os
import subprocess
from pathlib import Path
from typing import Dict, List

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

# Service definitions
SERVICES = {
    'stock': {
        'name': 'SentientStockMonitor',
        'display_name': 'Sentient Stock Informational Monitor',
        'script': 'stock_monitor_service.py',
        'description': 'Monitors stocks for trading opportunities'
    },
    'dex': {
        'name': 'SentientDEXLaunch',
        'display_name': 'Sentient DEX Launch Monitor',
        'script': 'dex_launch_service.py',
        'description': 'Monitors crypto DEX launches'
    },
    'crypto': {
        'name': 'SentientCryptoBreakout',
        'display_name': 'Sentient Crypto Breakout Monitor',
        'script': 'crypto_breakout_service.py',
        'description': 'Monitors crypto breakout patterns'
    }
}


def print_header(text: str):
    """Print section header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}\n")


def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


def print_info(text: str):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")


def run_service_command(service_key: str, command: str) -> bool:
    """
    Run a command on a service
    
    Args:
        service_key: Service identifier (stock, dex, crypto)
        command: Command to run (install, start, stop, remove, etc.)
    
    Returns:
        True if successful, False otherwise
    """
    if service_key not in SERVICES:
        print_error(f"Unknown service: {service_key}")
        return False
    
    service = SERVICES[service_key]
    script_path = Path(__file__).parent / service['script']
    
    if not script_path.exists():
        print_error(f"Service script not found: {script_path}")
        return False
    
    try:
        print_info(f"Running: {command} on {service['display_name']}...")
        
        # Run the service script with the command
        result = subprocess.run(
            [sys.executable, str(script_path), command],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print_success(f"{command.capitalize()} successful for {service['display_name']}")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            print_error(f"{command.capitalize()} failed for {service['display_name']}")
            if result.stderr:
                print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print_error(f"Command timed out for {service['display_name']}")
        return False
    except Exception as e:
        print_error(f"Error running command: {e}")
        return False


def get_service_status() -> Dict[str, str]:
    """
    Get status of all services
    
    Returns:
        Dictionary mapping service keys to status strings
    """
    statuses = {}
    
    try:
        # Use sc query to check service status
        for key, service in SERVICES.items():
            try:
                result = subprocess.run(
                    ['sc', 'query', service['name']],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.returncode == 0:
                    # Service exists, parse status
                    if 'RUNNING' in result.stdout:
                        statuses[key] = 'RUNNING'
                    elif 'STOPPED' in result.stdout:
                        statuses[key] = 'STOPPED'
                    else:
                        statuses[key] = 'UNKNOWN'
                else:
                    statuses[key] = 'NOT INSTALLED'
                    
            except Exception:
                statuses[key] = 'ERROR'
    
    except Exception as e:
        print_error(f"Error checking service status: {e}")
    
    return statuses


def show_status():
    """Show status of all services"""
    print_header("SENTIENT TRADER SERVICES STATUS")
    
    statuses = get_service_status()
    
    print(f"{'Service':<35} {'Status':<15} {'Description':<30}")
    print("-" * 80)
    
    for key, service in SERVICES.items():
        status = statuses.get(key, 'UNKNOWN')
        
        # Color code status
        if status == 'RUNNING':
            status_str = f"{Colors.GREEN}{status}{Colors.END}"
        elif status == 'STOPPED':
            status_str = f"{Colors.YELLOW}{status}{Colors.END}"
        elif status == 'NOT INSTALLED':
            status_str = f"{Colors.RED}{status}{Colors.END}"
        else:
            status_str = f"{Colors.RED}{status}{Colors.END}"
        
        print(f"{service['display_name']:<35} {status_str:<24} {service['description']:<30}")
    
    print()


def install_all():
    """Install all services"""
    print_header("INSTALLING ALL SERVICES")
    
    success_count = 0
    for key in SERVICES:
        if run_service_command(key, 'install'):
            success_count += 1
    
    print(f"\n{success_count}/{len(SERVICES)} services installed successfully")
    
    if success_count == len(SERVICES):
        print_success("All services installed! Use 'start-all' to start them.")
    else:
        print_warning("Some services failed to install. Check errors above.")


def uninstall_all():
    """Uninstall all services"""
    print_header("UNINSTALLING ALL SERVICES")
    
    # Stop all first
    print_info("Stopping all services first...")
    for key in SERVICES:
        run_service_command(key, 'stop')
    
    # Then uninstall
    success_count = 0
    for key in SERVICES:
        if run_service_command(key, 'remove'):
            success_count += 1
    
    print(f"\n{success_count}/{len(SERVICES)} services uninstalled successfully")


def start_all():
    """Start all services"""
    print_header("STARTING ALL SERVICES")
    
    success_count = 0
    for key in SERVICES:
        if run_service_command(key, 'start'):
            success_count += 1
    
    print(f"\n{success_count}/{len(SERVICES)} services started successfully")


def stop_all():
    """Stop all services"""
    print_header("STOPPING ALL SERVICES")
    
    success_count = 0
    for key in SERVICES:
        if run_service_command(key, 'stop'):
            success_count += 1
    
    print(f"\n{success_count}/{len(SERVICES)} services stopped successfully")


def show_help():
    """Show help message"""
    print(__doc__)
    
    print("\nInstalled Services:")
    statuses = get_service_status()
    for key, service in SERVICES.items():
        status = statuses.get(key, 'UNKNOWN')
        print(f"  {key:<10} - {service['display_name']} [{status}]")
    
    print("\nFor more information, see: docs/WINDOWS_SERVICES_GUIDE.md")


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        show_help()
        return
    
    command = sys.argv[1].lower()
    
    # Commands that operate on all services
    if command == 'install-all':
        install_all()
    elif command == 'uninstall-all':
        uninstall_all()
    elif command == 'start-all':
        start_all()
    elif command == 'stop-all':
        stop_all()
    elif command == 'status':
        show_status()
    
    # Commands that operate on specific service
    elif command in ['install', 'uninstall', 'start', 'stop', 'restart']:
        if len(sys.argv) < 3:
            print_error(f"Service name required for '{command}' command")
            print("Available services: stock, dex, crypto")
            return
        
        service_key = sys.argv[2].lower()
        
        if command == 'restart':
            run_service_command(service_key, 'stop')
            run_service_command(service_key, 'start')
        else:
            # Map command to service command
            cmd_map = {
                'install': 'install',
                'uninstall': 'remove',
                'start': 'start',
                'stop': 'stop'
            }
            run_service_command(service_key, cmd_map[command])
    
    else:
        print_error(f"Unknown command: {command}")
        show_help()


if __name__ == '__main__':
    # Check if running as administrator
    try:
        import ctypes
        is_admin = ctypes.windll.shell32.IsUserAnAdmin()
        if not is_admin:
            print_warning("This script should be run as Administrator for service management!")
            print_warning("Right-click and select 'Run as Administrator'\n")
    except:
        pass
    
    main()
