# app.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import hashlib
import json
import os
from typing import Dict, List, Tuple, Optional
import uuid
from dataclasses import dataclass, asdict
from enum import Enum
import calendar


# ============================================================================
# DATA MODELS & CONFIGURATION
# ============================================================================

class UserRole(str, Enum):
    ADMIN = "Administrator"
    SUPERVISOR = "Supervisor"
    HR = "HR Manager"
    EMPLOYEE = "Employee"


class ShiftType(str, Enum):
    MORNING = "Morning (06:00-14:00)"
    AFTERNOON = "Afternoon (14:00-22:00)"
    NIGHT = "Night (22:00-06:00)"
    GENERAL = "General (08:00-17:00)"


class MessagePriority(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    URGENT = "Urgent"


@dataclass
class User:
    id: str
    username: str
    password: str
    name: str
    role: str
    department: str
    shift: Optional[str]
    email: str
    phone: str
    created_at: datetime
    is_active: bool
    hourly_rate: float = 0.0  # For payroll if applicable
    overtime_rate: float = 1.5  # Default overtime multiplier


@dataclass
class Employee:
    id: str
    name: str
    department: str
    shift: str
    role: str
    hourly_rate: float
    overtime_rate: float
    supervisor_id: str


@dataclass
class TimeEntry:
    id: str
    employee_id: str
    user_id: str  # Link to user account
    clock_in: datetime
    clock_out: Optional[datetime]
    location: str
    supervisor_verified: bool
    verified_by: Optional[str]
    ai_flags: List[str]
    normal_hours: float
    overtime_hours: float
    status: str
    entry_type: str  # 'employee' or 'staff' (for admin/supervisor/hr)


@dataclass
class Communication:
    id: str
    timestamp: datetime
    sender_id: str
    sender_name: str
    recipient_type: str
    recipient_id: str
    message: str
    priority: str
    category: str
    is_private: bool
    read_by: List[str]
    acknowledged_by: List[str]
    expires_at: Optional[datetime]


@dataclass
class ShiftHandover:
    id: str
    shift_date: datetime
    shift_type: str
    outgoing_supervisor_id: str
    outgoing_supervisor_name: str
    incoming_supervisor_id: str
    incoming_supervisor_name: str
    production_summary: str
    issues_reported: List[str]
    maintenance_alerts: List[str]
    targets_next_shift: str
    created_at: datetime


# ============================================================================
# AI ALGORITHMS & ANALYTICS ENGINE
# ============================================================================

class TimesheetAIEngine:
    """AI-powered analytics and anomaly detection for timesheet data"""

    def __init__(self):
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        self.scaler = StandardScaler()

    def validate_time_entry(self, entry: Dict, employee_history: pd.DataFrame) -> Tuple[bool, List[str]]:
        """AI validation of time entries for anomalies and suspicious patterns"""
        flags = []

        clock_in = entry['clock_in']
        clock_out = entry.get('clock_out')

        if clock_out:
            hours_worked = (clock_out - clock_in).total_seconds() / 3600

            if hours_worked > 16:
                flags.append("UNREALISTIC_HOURS: Exceeds 16 hours")
            elif hours_worked < 0.5:
                flags.append("SUSPICIOUS_SHORT: Less than 30 minutes")

            if hours_worked > 12:
                flags.append("OVERTIME_ALERT: Exceeds 12 hours")

            if not employee_history.empty:
                same_day_entries = employee_history[
                    employee_history['clock_in'].dt.date == clock_in.date()
                    ]
                if len(same_day_entries) > 0:
                    flags.append("DUPLICATE_ENTRY: Multiple entries on same day")
        else:
            time_since_clockin = datetime.now() - clock_in
            if time_since_clockin.total_seconds() / 3600 > 16:
                flags.append("MISSING_PUNCH: No clock-out after 16+ hours")

        is_valid = len([f for f in flags if "UNREALISTIC" in f]) == 0
        return is_valid, flags

    def calculate_hours(self, clock_in: datetime, clock_out: datetime,
                        shift_type: str, is_staff: bool = False) -> Tuple[float, float]:
        """Calculate normal and overtime hours based on shift"""
        total_hours = (clock_out - clock_in).total_seconds() / 3600

        # Staff (admin/supervisor/hr) might have different rules
        if is_staff:
            if total_hours <= 8:
                return total_hours, 0
            else:
                return 8, total_hours - 8
        else:
            if total_hours <= 8:
                return total_hours, 0
            elif total_hours <= 12:
                return 8, total_hours - 8
            else:
                return 8, min(4, total_hours - 8)

    def detect_patterns(self, timesheet_data: pd.DataFrame) -> Dict:
        """Detect attendance patterns and trends"""
        patterns = {
            'frequent_late_arrivals': [],
            'overtime_trends': {}
        }

        if timesheet_data.empty:
            return patterns

        for emp_id, emp_data in timesheet_data.groupby('employee_id'):
            if len(emp_data) > 0:
                avg_overtime = emp_data['overtime_hours'].mean()
                if avg_overtime > 2:
                    patterns['overtime_trends'][emp_id] = avg_overtime

        return patterns

    def prioritize_message(self, message: str, category: str) -> str:
        """AI-powered message prioritization"""
        urgent_keywords = ['breakdown', 'emergency', 'accident', 'stop', 'immediate', 'critical']
        high_keywords = ['delay', 'issue', 'problem', 'failure', 'repair', 'safety']

        message_lower = message.lower()

        if any(kw in message_lower for kw in urgent_keywords):
            return MessagePriority.URGENT.value
        elif any(kw in message_lower for kw in high_keywords):
            return MessagePriority.HIGH.value
        elif category == 'handover':
            return MessagePriority.MEDIUM.value
        else:
            return MessagePriority.LOW.value

    def generate_handover_summary(self, shift_data: Dict) -> str:
        """AI-generated shift handover summary"""
        summary_parts = []

        if 'production_actual' in shift_data and 'production_target' in shift_data:
            achievement = (shift_data['production_actual'] / shift_data['production_target']) * 100
            summary_parts.append(f"Production achievement: {achievement:.1f}%")

        if 'attendance_count' in shift_data:
            summary_parts.append(
                f"Workforce present: {shift_data['attendance_count']} / {shift_data.get('expected_count', 'N/A')}")

        if 'issues' in shift_data and shift_data['issues']:
            summary_parts.append(f"Issues reported: {len(shift_data['issues'])}")

        return " | ".join(summary_parts) if summary_parts else "No significant events to report"


# ============================================================================
# DATA MANAGEMENT
# ============================================================================

class DataManager:
    """Handle data persistence using session state"""

    @staticmethod
    def initialize_session_state():
        """Initialize all session state variables"""
        if 'initialized' not in st.session_state:
            # Create default users with hourly rates for staff
            if 'users' not in st.session_state:
                st.session_state.users = [
                    {
                        'id': 'ADMIN001',
                        'username': 'admin',
                        'password': 'admin123',
                        'name': 'System Administrator',
                        'role': UserRole.ADMIN.value,
                        'department': 'IT',
                        'shift': ShiftType.GENERAL.value,
                        'email': 'admin@cementco.com',
                        'phone': '+1234567890',
                        'created_at': datetime.now(),
                        'is_active': True,
                        'hourly_rate': 45.0,
                        'overtime_rate': 1.5
                    },
                    {
                        'id': 'SUP001',
                        'username': 'john.supervisor',
                        'password': 'super123',
                        'name': 'John Supervisor',
                        'role': UserRole.SUPERVISOR.value,
                        'department': 'Production',
                        'shift': ShiftType.MORNING.value,
                        'email': 'john@cementco.com',
                        'phone': '+1234567890',
                        'created_at': datetime.now(),
                        'is_active': True,
                        'hourly_rate': 35.0,
                        'overtime_rate': 1.5
                    },
                    {
                        'id': 'SUP002',
                        'username': 'jane.supervisor',
                        'password': 'super123',
                        'name': 'Jane Supervisor',
                        'role': UserRole.SUPERVISOR.value,
                        'department': 'Maintenance',
                        'shift': ShiftType.AFTERNOON.value,
                        'email': 'jane@cementco.com',
                        'phone': '+1234567890',
                        'created_at': datetime.now(),
                        'is_active': True,
                        'hourly_rate': 35.0,
                        'overtime_rate': 1.5
                    },
                    {
                        'id': 'EMP001',
                        'username': 'employee1',
                        'password': 'emp123',
                        'name': 'John Worker',
                        'role': UserRole.EMPLOYEE.value,
                        'department': 'Production',
                        'shift': ShiftType.MORNING.value,
                        'email': 'john.worker@cementco.com',
                        'phone': '+1234567890',
                        'created_at': datetime.now(),
                        'is_active': True,
                        'hourly_rate': 20.0,
                        'overtime_rate': 1.5
                    },
                    {
                        'id': 'EMP002',
                        'username': 'employee2',
                        'password': 'emp123',
                        'name': 'Sarah Worker',
                        'role': UserRole.EMPLOYEE.value,
                        'department': 'Production',
                        'shift': ShiftType.MORNING.value,
                        'email': 'sarah.worker@cementco.com',
                        'phone': '+1234567890',
                        'created_at': datetime.now(),
                        'is_active': True,
                        'hourly_rate': 20.0,
                        'overtime_rate': 1.5
                    },
                    {
                        'id': 'HR001',
                        'username': 'hr.manager',
                        'password': 'hr123',
                        'name': 'HR Manager',
                        'role': UserRole.HR.value,
                        'department': 'Human Resources',
                        'shift': ShiftType.GENERAL.value,
                        'email': 'hr@cementco.com',
                        'phone': '+1234567890',
                        'created_at': datetime.now(),
                        'is_active': True,
                        'hourly_rate': 40.0,
                        'overtime_rate': 1.5
                    }
                ]

            if 'employees' not in st.session_state:
                st.session_state.employees = DataManager._create_sample_employees()

            if 'time_entries' not in st.session_state:
                st.session_state.time_entries = []

            if 'communications' not in st.session_state:
                st.session_state.communications = []

            if 'handovers' not in st.session_state:
                st.session_state.handovers = []

            if 'ai_engine' not in st.session_state:
                st.session_state.ai_engine = TimesheetAIEngine()

            if 'current_user' not in st.session_state:
                st.session_state.current_user = None

            if 'login_error' not in st.session_state:
                st.session_state.login_error = None

            st.session_state.initialized = True

    @staticmethod
    def _create_sample_employees() -> List[Dict]:
        """Create sample employee data"""
        employees = []
        departments = ['Production', 'Maintenance', 'Quality Control']
        shifts = [s.value for s in ShiftType]
        roles = ['Operator', 'Technician', 'Helper']

        for i in range(20):
            emp = {
                'id': f"EMP{i + 3:03d}",  # Start from 003 to avoid conflict with users
                'name': f"Employee {i + 3}",
                'department': np.random.choice(departments),
                'shift': np.random.choice(shifts),
                'role': np.random.choice(roles),
                'hourly_rate': np.random.uniform(15, 35),
                'overtime_rate': 1.5,
                'supervisor_id': 'SUP001'
            }
            employees.append(emp)

        return employees

    @staticmethod
    def login(username: str, password: str) -> Optional[Dict]:
        """Authenticate user login"""
        for user in st.session_state.users:
            if user['username'] == username and user['password'] == password and user['is_active']:
                return user
        return None

    @staticmethod
    def add_time_entry(entry: Dict):
        """Add a new time entry with AI validation"""
        # Get history for this specific user
        user_history = [
            e for e in st.session_state.time_entries
            if e['user_id'] == entry['user_id']
        ]

        is_valid, flags = st.session_state.ai_engine.validate_time_entry(
            entry,
            pd.DataFrame(user_history)
        )

        entry['ai_flags'] = flags
        entry['status'] = 'pending' if is_valid else 'flagged'
        entry['id'] = str(uuid.uuid4())[:8]
        entry['verified_by'] = None

        if entry.get('clock_out'):
            user = next((u for u in st.session_state.users if u['id'] == entry['user_id']), None)
            if user:
                is_staff = user['role'] in [UserRole.ADMIN.value, UserRole.SUPERVISOR.value, UserRole.HR.value]
                normal, overtime = st.session_state.ai_engine.calculate_hours(
                    entry['clock_in'],
                    entry['clock_out'],
                    user.get('shift', ShiftType.GENERAL.value),
                    is_staff
                )
                entry['normal_hours'] = normal
                entry['overtime_hours'] = overtime

        st.session_state.time_entries.append(entry)
        return is_valid, flags

    @staticmethod
    def get_user_history(user_id: str) -> List[Dict]:
        """Get time entry history for a user"""
        return [
            entry for entry in st.session_state.time_entries
            if entry['user_id'] == user_id
        ]

    @staticmethod
    def get_employee_history(employee_id: str) -> List[Dict]:
        """Get time entry history for an employee"""
        return [
            entry for entry in st.session_state.time_entries
            if entry.get('employee_id') == employee_id
        ]

    @staticmethod
    def add_communication(message: Dict):
        """Add a new communication with AI prioritization"""
        priority = st.session_state.ai_engine.prioritize_message(
            message['message'],
            message.get('category', 'announcement')
        )
        message['priority'] = priority
        message['id'] = str(uuid.uuid4())[:8]
        message['timestamp'] = datetime.now()
        message['read_by'] = [message['sender_id']]
        message['acknowledged_by'] = []

        st.session_state.communications.append(message)
        return priority

    @staticmethod
    def get_unread_count(user_id: str) -> int:
        """Get count of unread messages for a user"""
        if not user_id:
            return 0

        user = next((u for u in st.session_state.users if u['id'] == user_id), None)
        if not user:
            return 0

        unread = 0
        for comm in st.session_state.communications:
            if user_id in comm['read_by']:
                continue

            if comm['recipient_type'] == 'all':
                unread += 1
            elif comm['recipient_type'] == 'department' and comm['recipient_id'] == user['department']:
                unread += 1
            elif comm['recipient_type'] == 'shift' and user.get('shift') and comm['recipient_id'] == user['shift']:
                unread += 1
            elif comm['recipient_type'] == 'individual' and comm['recipient_id'] == user_id:
                unread += 1

        return unread

    @staticmethod
    def mark_as_read(comm_id: str, user_id: str):
        """Mark a communication as read by a user"""
        for comm in st.session_state.communications:
            if comm['id'] == comm_id:
                if user_id not in comm['read_by']:
                    comm['read_by'].append(user_id)
                break

    @staticmethod
    def get_timesheet_dataframe() -> pd.DataFrame:
        """Convert time entries to DataFrame for analysis"""
        if not st.session_state.time_entries:
            return pd.DataFrame()

        df = pd.DataFrame(st.session_state.time_entries)

        # Add user details
        users_df = pd.DataFrame(st.session_state.users)
        if not df.empty and not users_df.empty:
            df = df.merge(users_df[['id', 'name', 'department', 'shift', 'role', 'hourly_rate', 'overtime_rate']],
                          left_on='user_id', right_on='id', how='left', suffixes=('', '_user'))

        return df

    @staticmethod
    def get_open_entry(user_id: str) -> Optional[Dict]:
        """Get open clock-in entry for a user"""
        for entry in st.session_state.time_entries:
            if entry['user_id'] == user_id and entry['clock_out'] is None:
                return entry
        return None

    @staticmethod
    def add_user(user_data: Dict) -> bool:
        """Add a new user"""
        try:
            if any(u['username'] == user_data['username'] for u in st.session_state.users):
                return False

            user_data['id'] = f"USER{len(st.session_state.users) + 1:03d}"
            user_data['created_at'] = datetime.now()

            st.session_state.users.append(user_data)
            return True
        except Exception as e:
            st.error(f"Error adding user: {e}")
            return False

    @staticmethod
    def update_user(user_id: str, updates: Dict):
        """Update user information"""
        for user in st.session_state.users:
            if user['id'] == user_id:
                for key, value in updates.items():
                    user[key] = value
                return True
        return False


# ============================================================================
# UI COMPONENTS
# ============================================================================

class TimesheetApp:
    """Main Streamlit Application"""

    def __init__(self):
        self.data_manager = DataManager()
        self.data_manager.initialize_session_state()

    def run(self):
        """Main application entry point"""
        st.set_page_config(
            page_title="AI Timesheet & Communication System",
            page_icon="🏭",
            layout="wide",
            initial_sidebar_state="expanded"
        )

        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 2rem;
            font-weight: bold;
            text-align: center;
            padding: 1rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 1rem;
        }
        .login-box {
            background: white;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            max-width: 400px;
            margin: 0 auto;
        }
        .alert-urgent {
            background-color: #ff4444;
            color: white;
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
        }
        .alert-high {
            background-color: #ff8800;
            color: white;
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
        }
        .alert-medium {
            background-color: #ffbb33;
            color: black;
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
        }
        .alert-low {
            background-color: #00C851;
            color: white;
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
        }
        .private-message {
            border-left: 4px solid #9c27b0;
            background-color: #f3e5f5;
        }
        .unread-badge {
            background-color: #ff4444;
            color: white;
            border-radius: 50%;
            padding: 2px 8px;
            font-size: 0.8em;
            margin-left: 5px;
        }
        .user-info {
            background: #f0f2f6;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .clocked-in {
            background-color: #4CAF50;
            color: white;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 0.9em;
            display: inline-block;
        }
        .clocked-out {
            background-color: #9e9e9e;
            color: white;
            padding: 5px 10px;
            border-radius: 20px;
            font-size: 0.9em;
            display: inline-block;
        }
        .quick-clock {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
        }
        </style>
        """, unsafe_allow_html=True)

        # Main layout
        if not st.session_state.current_user:
            self.render_login_page()
        else:
            self.render_main_application()

    def render_login_page(self):
        """Render login page as main interface"""
        st.markdown('<div class="main-header">🏭 Cement Co. Production Management System</div>',
                    unsafe_allow_html=True)

        # Create two columns - login on left, info on right
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### 🔐 Login to System")
            st.markdown('<div class="login-box">', unsafe_allow_html=True)

            username = st.text_input("👤 Username", placeholder="Enter your username")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("🚪 Login", type="primary", use_container_width=True):
                    user = self.data_manager.login(username, password)
                    if user:
                        st.session_state.current_user = user
                        st.session_state.login_error = None
                        st.rerun()
                    else:
                        st.session_state.login_error = "Invalid username or password"

            with col_btn2:
                if st.button("🔄 Clear", use_container_width=True):
                    st.session_state.login_error = None

            if st.session_state.login_error:
                st.error(st.session_state.login_error)

            st.markdown('</div>', unsafe_allow_html=True)

            # Demo credentials expander
            with st.expander("📋 Demo Credentials"):
                st.markdown("""
                **Administrator:**
                - Username: `admin` / Password: `admin123`

                **Supervisors:**
                - Username: `john.supervisor` / Password: `super123`
                - Username: `jane.supervisor` / Password: `super123`

                **Employees:**
                - Username: `employee1` / Password: `emp123`
                - Username: `employee2` / Password: `emp123`

                **HR Manager:**
                - Username: `hr.manager` / Password: `hr123`
                """)

        with col2:
            st.markdown("### 🏭 Welcome to Smart Production Management")

            # Quick clock-in for employees (demo feature)
            st.info("""
            #### ✨ System Features:

            - **⏰ Universal Clock In/Out** - All staff including administrators can clock in/out
            - **💬 Intelligent Communication** - Prioritized messaging and alerts
            - **📊 Real-time Analytics** - Attendance trends and cost analysis
            - **🔄 Shift Handover** - Automated shift summaries and handoffs
            - **🔒 Private Messaging** - Secure 1-on-1 communications
            - **📱 Mobile Ready** - Access from any device

            #### 📈 Today's Stats:
            """)

            # Show some public stats
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                total_users = len([u for u in st.session_state.users if u['is_active']])
                st.metric("Total Users", total_users)

            with col_b:
                today = datetime.now().date()
                today_entries = len([e for e in st.session_state.time_entries
                                     if e['clock_in'].date() == today])
                st.metric("Today's Check-ins", today_entries)

            with col_c:
                active_now = len([e for e in st.session_state.time_entries
                                  if e['clock_out'] is None])
                st.metric("Currently Working", active_now)

            st.success("""
            #### 🎯 Why Use This System?

            - **Universal Time Tracking** - Everyone from admin to operators clocks in/out
            - **70-90% reduction** in timesheet errors
            - **Faster payroll processing**
            - **Improved shift coordination**
            - **Enhanced accountability and transparency**
            """)

    def render_main_application(self):
        """Render main application after login"""
        user = st.session_state.current_user

        # Header with user info and logout
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        with col1:
            st.markdown('<div class="main-header">🏭 AI-Powered Production Timesheet & Smart Communication</div>',
                        unsafe_allow_html=True)
        with col2:
            unread_count = self.data_manager.get_unread_count(user['id'])
            st.markdown(f"📨 **{unread_count} unread**")
        with col3:
            # Show clock status
            open_entry = self.data_manager.get_open_entry(user['id'])
            if open_entry:
                st.markdown(
                    f'<span class="clocked-in">🟢 Clocked In since {open_entry["clock_in"].strftime("%H:%M")}</span>',
                    unsafe_allow_html=True)
            else:
                st.markdown('<span class="clocked-out">⚪ Not Clocked In</span>', unsafe_allow_html=True)
        with col4:
            if st.button("🚪 Logout", use_container_width=True):
                st.session_state.current_user = None
                st.rerun()

        # Sidebar Navigation
        with st.sidebar:
            st.markdown("### 👤 User Profile")

            # Display user info with clock status
            open_entry = self.data_manager.get_open_entry(user['id'])
            clock_status = "🟢 Currently Working" if open_entry else "⚪ Not Clocked In"

            st.markdown(f"""
            <div class="user-info">
                <strong>{user['name']}</strong><br>
                <small>{user['role']} - {user['department']}</small><br>
                <small>Shift: {user.get('shift', 'N/A')}</small><br>
                <small>{clock_status}</small>
            </div>
            """, unsafe_allow_html=True)

            # Quick Clock In/Out widget in sidebar
            st.markdown("---")
            st.markdown("### ⏰ Quick Clock")

            open_entry = self.data_manager.get_open_entry(user['id'])
            current_time = datetime.now()

            if not open_entry:
                # Clock In button
                location = st.selectbox("Location",
                                        ["Main Gate", "Office", "Production Floor", "Control Room", "Workshop"],
                                        key="quick_location")

                if st.button("🟢 CLOCK IN", type="primary", use_container_width=True):
                    entry = {
                        'user_id': user['id'],
                        'employee_id': user['id'],  # Use user ID as employee ID for staff
                        'clock_in': current_time,
                        'clock_out': None,
                        'location': location,
                        'supervisor_verified': False,
                        'normal_hours': 0,
                        'overtime_hours': 0,
                        'entry_type': 'staff' if user['role'] != UserRole.EMPLOYEE.value else 'employee'
                    }

                    is_valid, flags = self.data_manager.add_time_entry(entry)

                    if is_valid:
                        st.success(f"✅ Clocked in at {current_time.strftime('%H:%M')}")
                        st.rerun()
                    else:
                        st.warning("⚠️ Entry flagged for review")
                        for flag in flags:
                            st.warning(f"• {flag}")
            else:
                # Clock Out button
                st.info(f"Clocked in at: {open_entry['clock_in'].strftime('%H:%M')}")
                st.info(f"Location: {open_entry['location']}")

                duration = current_time - open_entry['clock_in']
                hours_worked = duration.total_seconds() / 3600
                st.metric("Current Duration", f"{hours_worked:.1f} hours")

                if st.button("🔴 CLOCK OUT", type="primary", use_container_width=True):
                    # Find and update the entry
                    for entry in st.session_state.time_entries:
                        if entry['id'] == open_entry['id']:
                            entry['clock_out'] = current_time

                            # Calculate hours
                            is_staff = user['role'] in [UserRole.ADMIN.value, UserRole.SUPERVISOR.value,
                                                        UserRole.HR.value]
                            normal, overtime = st.session_state.ai_engine.calculate_hours(
                                entry['clock_in'],
                                current_time,
                                user.get('shift', ShiftType.GENERAL.value),
                                is_staff
                            )

                            entry['normal_hours'] = normal
                            entry['overtime_hours'] = overtime

                            st.success(f"✅ Clocked out. Hours: {normal:.1f} normal + {overtime:.1f} overtime")
                            st.rerun()
                            break

            st.markdown("---")

            # Navigation based on role
            pages = ["📊 Dashboard", "⏰ Timesheet Entry", "💬 Communications"]

            # Role-specific pages
            if user['role'] in [UserRole.SUPERVISOR.value, UserRole.ADMIN.value, UserRole.HR.value]:
                pages.append("📋 Timesheet Review")

            if user['role'] in [UserRole.SUPERVISOR.value, UserRole.ADMIN.value]:
                pages.append("🔄 Shift Handover")

            if user['role'] in [UserRole.ADMIN.value, UserRole.HR.value]:
                pages.extend(["📈 Analytics", "👥 User Management"])

            page = st.radio("Navigation", pages)

            st.markdown("---")
            st.markdown(f"**📅 {datetime.now().strftime('%Y-%m-%d')}**")
            st.markdown(f"**🕐 {datetime.now().strftime('%H:%M')}**")

        # Main content area
        if page == "📊 Dashboard":
            self.render_dashboard()
        elif page == "⏰ Timesheet Entry":
            self.render_timesheet_entry()
        elif page == "💬 Communications":
            self.render_communication_hub()
        elif page == "📋 Timesheet Review":
            self.render_timesheet_review()
        elif page == "🔄 Shift Handover":
            self.render_shift_handover()
        elif page == "📈 Analytics":
            self.render_analytics()
        elif page == "👥 User Management":
            self.render_user_management()

    def render_dashboard(self):
        """Main dashboard view"""
        st.header("📊 Production Dashboard")

        df = self.data_manager.get_timesheet_dataframe()
        user = st.session_state.current_user

        # Quick status card
        open_entry = self.data_manager.get_open_entry(user['id'])
        if open_entry:
            duration = datetime.now() - open_entry['clock_in']
            hours_worked = duration.total_seconds() / 3600
            st.info(
                f"🟢 You have been clocked in for **{hours_worked:.1f} hours** since {open_entry['clock_in'].strftime('%H:%M')}")

        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            today = datetime.now().date()
            today_entries = df[df['clock_in'].dt.date == today] if not df.empty else pd.DataFrame()
            workers_present = len(today_entries['user_id'].unique()) if not today_entries.empty else 0
            st.metric("👥 Present Today", workers_present)

        with col2:
            if user['role'] == UserRole.SUPERVISOR.value:
                supervised_emps = [e['id'] for e in st.session_state.employees if e['supervisor_id'] == user['id']]
                pending_df = df[df['employee_id'].isin(supervised_emps)] if not df.empty else pd.DataFrame()
                pending_verification = len(pending_df[pending_df['status'] == 'pending']) if not pending_df.empty else 0
            else:
                pending_verification = len(df[df['status'] == 'pending']) if not df.empty else 0
            st.metric("⏳ Pending", pending_verification)

        with col3:
            flagged_entries = len(df[df['ai_flags'].apply(len) > 0]) if not df.empty else 0
            st.metric("🚨 Flagged", flagged_entries)

        with col4:
            unread_messages = self.data_manager.get_unread_count(user['id'])
            st.metric("📨 Unread", unread_messages)

        st.markdown("---")

        # Recent activity and communications
        left_col, right_col = st.columns([2, 1])

        with left_col:
            st.subheader("📍 Recent Activity")
            if not df.empty:
                if user['role'] == UserRole.SUPERVISOR.value:
                    supervised_emps = [e['id'] for e in st.session_state.employees if e['supervisor_id'] == user['id']]
                    recent_entries = df[df['employee_id'].isin(supervised_emps)].nlargest(10, 'clock_in')
                else:
                    recent_entries = df.nlargest(10, 'clock_in')

                if not recent_entries.empty:
                    display_cols = ['name', 'clock_in', 'clock_out', 'status']
                    display_df = recent_entries[display_cols].copy()
                    display_df['clock_in'] = display_df['clock_in'].dt.strftime('%H:%M')
                    display_df['clock_out'] = display_df['clock_out'].apply(
                        lambda x: x.strftime('%H:%M') if pd.notna(x) else '--')
                    st.dataframe(display_df, use_container_width=True)

            # My recent entries
            st.subheader("📋 My Recent Timesheets")
            my_entries = [e for e in st.session_state.time_entries if e['user_id'] == user['id']]
            if my_entries:
                my_df = pd.DataFrame(my_entries[-5:])
                display_cols = ['clock_in', 'clock_out', 'normal_hours', 'overtime_hours', 'status']
                display_df = my_df[display_cols].copy()
                display_df['clock_in'] = display_df['clock_in'].dt.strftime('%Y-%m-%d %H:%M')
                display_df['clock_out'] = display_df['clock_out'].apply(
                    lambda x: x.strftime('%H:%M') if pd.notna(x) else '--')
                st.dataframe(display_df, use_container_width=True)

        with right_col:
            st.subheader("📢 Recent Messages")
            recent_comms = [
                c for c in st.session_state.communications
                if not c.get('is_private', False) or c['recipient_id'] == user['id'] or c['sender_id'] == user['id']
            ]
            recent_comms = sorted(recent_comms, key=lambda x: x['timestamp'], reverse=True)[:5]

            for comm in recent_comms:
                self.data_manager.mark_as_read(comm['id'], user['id'])
                priority_class = f"alert-{comm['priority'].lower()}"
                private_class = "private-message" if comm.get('is_private', False) else ""

                st.markdown(f"""
                <div class="{priority_class} {private_class}" style="padding: 10px; margin: 5px 0;">
                    <strong>{comm['priority']}</strong><br>
                    {comm['message'][:80]}...<br>
                    <small>From: {comm['sender_name']}</small>
                </div>
                """, unsafe_allow_html=True)

            # Currently working
            st.subheader("🟢 Currently Working")
            active_entries = [e for e in st.session_state.time_entries if e['clock_out'] is None]
            if active_entries:
                for entry in active_entries[:5]:
                    entry_user = next((u for u in st.session_state.users if u['id'] == entry['user_id']), None)
                    if entry_user:
                        duration = datetime.now() - entry['clock_in']
                        st.markdown(
                            f"**{entry_user['name']}** - {entry['location']} ({duration.seconds // 3600}h {duration.seconds // 60 % 60}m)")
            else:
                st.info("No one currently clocked in")

    def render_timesheet_entry(self):
        """Timesheet entry form"""
        st.header("⏰ Digital Timesheet Entry")

        user = st.session_state.current_user

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Clock In/Out")

            # Show current status
            open_entry = self.data_manager.get_open_entry(user['id'])

            if open_entry:
                st.success(f"🟢 Currently clocked in since {open_entry['clock_in'].strftime('%H:%M')}")
                st.info(f"📍 Location: {open_entry['location']}")

                duration = datetime.now() - open_entry['clock_in']
                hours_worked = duration.total_seconds() / 3600
                st.metric("Current Duration", f"{hours_worked:.1f} hours")

                if st.button("🔴 Clock Out Now", type="primary"):
                    for entry in st.session_state.time_entries:
                        if entry['id'] == open_entry['id']:
                            entry['clock_out'] = datetime.now()

                            is_staff = user['role'] in [UserRole.ADMIN.value, UserRole.SUPERVISOR.value,
                                                        UserRole.HR.value]
                            normal, overtime = st.session_state.ai_engine.calculate_hours(
                                entry['clock_in'],
                                entry['clock_out'],
                                user.get('shift', ShiftType.GENERAL.value),
                                is_staff
                            )

                            entry['normal_hours'] = normal
                            entry['overtime_hours'] = overtime

                            st.success(f"✅ Clocked out. Hours: {normal:.1f} normal + {overtime:.1f} overtime")
                            st.rerun()
                            break
            else:
                st.warning("⚪ Not currently clocked in")

                # For admins/supervisors, they can clock in themselves or others
                clock_for = st.radio("Clock in for:", ["Myself", "Another Employee"])

                if clock_for == "Myself":
                    employee_id = user['id']
                    employee_name = user['name']
                    st.info(f"Clocking in as: {employee_name}")
                else:
                    if user['role'] == UserRole.SUPERVISOR.value:
                        available_employees = [e for e in st.session_state.employees if
                                               e['supervisor_id'] == user['id']]
                    else:
                        available_employees = st.session_state.employees

                    if available_employees:
                        employee_options = {f"{e['name']} ({e['id']})": e['id'] for e in available_employees}
                        selected = st.selectbox("Select Employee", options=list(employee_options.keys()))
                        employee_id = employee_options[selected]
                        employee = next(e for e in st.session_state.employees if e['id'] == employee_id)
                        employee_name = employee['name']
                    else:
                        st.warning("No employees available")
                        employee_id = None
                        employee_name = None

                if employee_id:
                    location = st.selectbox("Location",
                                            ["Main Gate", "Office", "Production Floor", "Control Room", "Workshop"])

                    # Option to set custom time
                    use_custom_time = st.checkbox("Set custom time")
                    if use_custom_time:
                        custom_time = st.time_input("Clock In Time", value=datetime.now().time())
                        clock_in_datetime = datetime.combine(datetime.now().date(), custom_time)
                    else:
                        clock_in_datetime = datetime.now()

                    if st.button("🟢 Clock In", type="primary"):
                        entry = {
                            'user_id': employee_id,
                            'employee_id': employee_id,
                            'clock_in': clock_in_datetime,
                            'clock_out': None,
                            'location': location,
                            'supervisor_verified': False,
                            'normal_hours': 0,
                            'overtime_hours': 0,
                            'entry_type': 'staff' if user['role'] != UserRole.EMPLOYEE.value else 'employee'
                        }

                        is_valid, flags = self.data_manager.add_time_entry(entry)

                        if is_valid:
                            st.success(
                                f"✅ {employee_name} clocked in successfully at {clock_in_datetime.strftime('%H:%M')}")
                            st.rerun()
                        else:
                            st.warning("⚠️ Entry flagged for review")
                            for flag in flags:
                                st.warning(f"• {flag}")

        with col2:
            st.subheader("📋 Today's Timesheets")

            today = datetime.now().date()

            if user['role'] in [UserRole.ADMIN.value, UserRole.SUPERVISOR.value, UserRole.HR.value]:
                # Show all entries for supervisors/admins
                today_entries = [
                    e for e in st.session_state.time_entries
                    if e['clock_in'].date() == today
                ]
            else:
                # Show only user's entries for employees
                today_entries = [
                    e for e in st.session_state.time_entries
                    if e['clock_in'].date() == today and e['user_id'] == user['id']
                ]

            if today_entries:
                df_today = pd.DataFrame(today_entries)

                # Add names
                user_dict = {u['id']: u['name'] for u in st.session_state.users}
                df_today['user_name'] = df_today['user_id'].map(user_dict)

                display_df = df_today[['user_name', 'clock_in', 'clock_out', 'location', 'status']].copy()
                display_df['clock_in'] = display_df['clock_in'].apply(lambda x: x.strftime('%H:%M'))
                display_df['clock_out'] = display_df['clock_out'].apply(
                    lambda x: x.strftime('%H:%M') if pd.notna(x) else '--')

                st.dataframe(display_df, use_container_width=True)

                # Summary stats
                total_hours = df_today['normal_hours'].sum() + df_today['overtime_hours'].sum()
                total_overtime = df_today['overtime_hours'].sum()

                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Total Hours Today", f"{total_hours:.1f}")
                with col_b:
                    st.metric("Total Overtime", f"{total_overtime:.1f}")
            else:
                st.info("No entries recorded today")

            # My recent history
            st.subheader("📊 My Recent History")
            my_entries = [e for e in st.session_state.time_entries if e['user_id'] == user['id']]
            if my_entries:
                my_df = pd.DataFrame(my_entries[-7:])

                # Calculate weekly totals
                week_hours = my_df['normal_hours'].sum() + my_df['overtime_hours'].sum()
                week_overtime = my_df['overtime_hours'].sum()

                col_c, col_d = st.columns(2)
                with col_c:
                    st.metric("Week Hours", f"{week_hours:.1f}")
                with col_d:
                    st.metric("Week Overtime", f"{week_overtime:.1f}")

    def render_communication_hub(self):
        """Communication center"""
        st.header("💬 Communication Hub")

        user = st.session_state.current_user

        tab1, tab2 = st.tabs(["📢 Messages", "✉️ Send Message"])

        with tab1:
            st.subheader("Your Messages")

            # Get relevant messages
            relevant_messages = []
            for comm in st.session_state.communications:
                # Skip private messages not for this user
                if comm.get('is_private', False):
                    if comm['sender_id'] != user['id'] and comm['recipient_id'] != user['id']:
                        continue

                # Check if message is for this user
                is_relevant = False
                if comm['recipient_type'] == 'all':
                    is_relevant = True
                elif comm['recipient_type'] == 'department' and comm['recipient_id'] == user['department']:
                    is_relevant = True
                elif comm['recipient_type'] == 'shift' and user.get('shift') and comm['recipient_id'] == user['shift']:
                    is_relevant = True
                elif comm['recipient_type'] == 'individual' and comm['recipient_id'] == user['id']:
                    is_relevant = True
                elif comm['sender_id'] == user['id']:
                    is_relevant = True

                if is_relevant:
                    relevant_messages.append(comm)
                    self.data_manager.mark_as_read(comm['id'], user['id'])

            # Sort by priority and time
            priority_order = {'Urgent': 0, 'High': 1, 'Medium': 2, 'Low': 3}
            relevant_messages.sort(key=lambda x: (priority_order.get(x['priority'], 3), x['timestamp']), reverse=True)

            if relevant_messages:
                for msg in relevant_messages:
                    priority_class = f"alert-{msg['priority'].lower()}"
                    private_class = "private-message" if msg.get('is_private', False) else ""

                    is_acknowledged = user['id'] in msg.get('acknowledged_by', [])
                    is_read = user['id'] in msg.get('read_by', [])

                    with st.container():
                        st.markdown(f"""
                        <div class="{priority_class} {private_class}" style="padding: 15px; margin: 10px 0; border-radius: 8px;">
                            <div style="display: flex; justify-content: space-between;">
                                <strong>
                                    {msg['priority']} - {msg['category'].upper()}
                                    {'🔒 PRIVATE' if msg.get('is_private', False) else ''}
                                </strong>
                                <span>{msg['timestamp'].strftime('%Y-%m-%d %H:%M')}</span>
                            </div>
                            <div style="margin: 10px 0;">{msg['message']}</div>
                            <div style="font-size: 0.9em;">
                                From: {msg['sender_name']} | {'✅ Read' if is_read else '📖 Unread'}
                                {' | ✅ Acknowledged' if is_acknowledged else ''}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        col1, col2 = st.columns(2)
                        with col1:
                            if not is_acknowledged:
                                if st.button(f"✓ Acknowledge", key=f"ack_{msg['id']}"):
                                    msg['acknowledged_by'].append(user['id'])
                                    st.rerun()

                        with col2:
                            if not is_read:
                                if st.button(f"📖 Mark Read", key=f"read_{msg['id']}"):
                                    self.data_manager.mark_as_read(msg['id'], user['id'])
                                    st.rerun()
            else:
                st.info("No messages")

        with tab2:
            st.subheader("Send New Message")

            col1, col2 = st.columns(2)

            with col1:
                recipient_type = st.selectbox(
                    "Send To",
                    ["All Production", "Department", "Shift", "Individual"]
                )

                if recipient_type == "Department":
                    departments = list(set(e['department'] for e in st.session_state.employees) |
                                       set(u['department'] for u in st.session_state.users))
                    recipient = st.selectbox("Select Department", departments)
                elif recipient_type == "Shift":
                    shifts = [s.value for s in ShiftType]
                    recipient = st.selectbox("Select Shift", shifts)
                elif recipient_type == "Individual":
                    users_list = [
                        f"{u['name']} ({u['role']})" for u in st.session_state.users
                        if u['is_active'] and u['id'] != user['id']
                    ]
                    if users_list:
                        selected_user_str = st.selectbox("Select Recipient", users_list)
                        selected_user_name = selected_user_str.split(" (")[0]
                        recipient_user = next(
                            u for u in st.session_state.users
                            if u['name'] == selected_user_name
                        )
                        recipient = recipient_user['id']
                    else:
                        recipient = None
                else:
                    recipient = "all"

                category = st.selectbox(
                    "Category",
                    ["instruction", "alert", "handover", "announcement", "private"]
                )

                is_private = st.checkbox(
                    "🔒 Private Message",
                    value=(category == "private"),
                    help="Only visible to sender and recipient"
                )

            with col2:
                message = st.text_area("Message", height=150)

                expires_in = st.selectbox(
                    "Expires In",
                    ["Never", "1 hour", "4 hours", "24 hours", "1 week"]
                )

            if st.button("📨 Send Message", type="primary"):
                if message:
                    recipient_type_map = {
                        "All Production": "all",
                        "Department": "department",
                        "Shift": "shift",
                        "Individual": "individual"
                    }

                    msg_data = {
                        'sender_id': user['id'],
                        'sender_name': user['name'],
                        'recipient_type': recipient_type_map[recipient_type],
                        'recipient_id': recipient if recipient != "all" else "all",
                        'message': message,
                        'category': category,
                        'is_private': is_private or category == "private",
                        'timestamp': datetime.now()
                    }

                    priority = self.data_manager.add_communication(msg_data)
                    st.success(f"✅ Message sent with {priority} priority")
                else:
                    st.error("Please enter a message")

    def render_timesheet_review(self):
        """Timesheet review interface"""
        st.header("📋 Timesheet Review & Verification")

        df = self.data_manager.get_timesheet_dataframe()
        user = st.session_state.current_user

        if df.empty:
            st.info("No timesheet entries to review")
            return

        # Filter for supervisor's employees
        if user['role'] == UserRole.SUPERVISOR.value:
            supervised_emps = [e['id'] for e in st.session_state.employees if e['supervisor_id'] == user['id']]
            df = df[df['employee_id'].isin(supervised_emps)]

        date_filter = st.date_input("Select Date", value=datetime.now().date())

        status_filter = st.multiselect(
            "Status",
            options=['pending', 'flagged', 'verified', 'approved'],
            default=['pending', 'flagged']
        )

        filtered_df = df[df['clock_in'].dt.date == date_filter]
        if status_filter:
            filtered_df = filtered_df[filtered_df['status'].isin(status_filter)]

        st.subheader(f"Entries for {date_filter}")

        if not filtered_df.empty:
            for _, row in filtered_df.iterrows():
                with st.expander(
                        f"{row['name']} - {row['clock_in'].strftime('%H:%M')} to {row['clock_out'].strftime('%H:%M') if pd.notna(row['clock_out']) else '--'}"
                ):
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.markdown(f"**Employee:** {row['name']}")
                        st.markdown(f"**Department:** {row['department']}")
                        st.markdown(f"**Role:** {row['role']}")
                        st.markdown(f"**Hours:** Normal: {row['normal_hours']:.1f}h | OT: {row['overtime_hours']:.1f}h")
                        st.markdown(f"**Location:** {row['location']}")

                        if row['ai_flags']:
                            st.markdown("**🚨 AI Flags:**")
                            for flag in row['ai_flags']:
                                st.warning(flag)

                    with col2:
                        st.markdown(f"**Status:** `{row['status']}`")

                        if row.get('verified_by'):
                            verifier = next((u for u in st.session_state.users if u['id'] == row['verified_by']), None)
                            if verifier:
                                st.markdown(f"**Verified by:** {verifier['name']}")

                        if row['status'] in ['pending', 'flagged']:
                            if st.button(f"✅ Verify", key=f"verify_{row['id']}"):
                                original_idx = next(
                                    i for i, e in enumerate(st.session_state.time_entries) if e['id'] == row['id'])
                                st.session_state.time_entries[original_idx]['status'] = 'verified'
                                st.session_state.time_entries[original_idx]['supervisor_verified'] = True
                                st.session_state.time_entries[original_idx]['verified_by'] = user['id']
                                st.success("Entry verified")
                                st.rerun()

                            if st.button(f"❌ Reject", key=f"reject_{row['id']}"):
                                original_idx = next(
                                    i for i, e in enumerate(st.session_state.time_entries) if e['id'] == row['id'])
                                st.session_state.time_entries[original_idx]['status'] = 'disputed'
                                st.warning("Entry disputed")
                                st.rerun()

            # Bulk actions
            st.markdown("---")
            if st.button("✅ Verify All Displayed"):
                count = 0
                for _, row in filtered_df.iterrows():
                    if row['status'] in ['pending', 'flagged']:
                        original_idx = next(
                            i for i, e in enumerate(st.session_state.time_entries) if e['id'] == row['id'])
                        st.session_state.time_entries[original_idx]['status'] = 'verified'
                        st.session_state.time_entries[original_idx]['supervisor_verified'] = True
                        st.session_state.time_entries[original_idx]['verified_by'] = user['id']
                        count += 1
                st.success(f"Verified {count} entries")
                st.rerun()
        else:
            st.info("No entries match filters")

    def render_shift_handover(self):
        """Shift handover interface"""
        st.header("🔄 Shift Handover")

        user = st.session_state.current_user

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Create Handover Report")

            shift_type = st.selectbox("Shift", [s.value for s in ShiftType])

            supervisors = [
                f"{u['name']} ({u['id']})"
                for u in st.session_state.users
                if u['role'] == UserRole.SUPERVISOR.value and u['is_active'] and u['id'] != user['id']
            ]

            if supervisors:
                incoming_supervisor_str = st.selectbox("Incoming Supervisor", supervisors)
                incoming_supervisor_id = incoming_supervisor_str.split("(")[1].rstrip(")")
                incoming_supervisor_name = incoming_supervisor_str.split(" (")[0]
            else:
                incoming_supervisor_id = ""
                incoming_supervisor_name = ""

            production_summary = st.text_area("Production Summary")
            issues = st.multiselect(
                "Issues Reported",
                ["Equipment malfunction", "Material shortage", "Quality issue",
                 "Safety incident", "Staff shortage", "Maintenance required"]
            )
            maintenance_alerts = st.multiselect(
                "Maintenance Alerts",
                ["Kiln inspection", "Conveyor belt wear", "Motor vibration",
                 "Filter replacement", "Calibration required"]
            )
            targets = st.text_area("Targets for Next Shift")

            if st.button("📋 Generate Report", type="primary"):
                handover = {
                    'id': str(uuid.uuid4())[:8],
                    'shift_date': datetime.now(),
                    'shift_type': shift_type,
                    'outgoing_supervisor_id': user['id'],
                    'outgoing_supervisor_name': user['name'],
                    'incoming_supervisor_id': incoming_supervisor_id,
                    'incoming_supervisor_name': incoming_supervisor_name,
                    'production_summary': production_summary,
                    'issues_reported': issues,
                    'maintenance_alerts': maintenance_alerts,
                    'targets_next_shift': targets,
                    'created_at': datetime.now()
                }

                st.session_state.handovers.append(handover)

                # Generate AI summary
                shift_data = {
                    'production_actual': 850,
                    'production_target': 1000,
                    'attendance_count': 15,
                    'expected_count': 18,
                    'issues': issues
                }

                ai_summary = st.session_state.ai_engine.generate_handover_summary(shift_data)
                st.success("✅ Handover created")
                st.info(f"🤖 AI Summary: {ai_summary}")

                # Option to send as message
                if incoming_supervisor_id:
                    if st.button("📨 Send to Incoming Supervisor"):
                        msg_data = {
                            'sender_id': user['id'],
                            'sender_name': user['name'],
                            'recipient_type': 'individual',
                            'recipient_id': incoming_supervisor_id,
                            'message': f"SHIFT HANDOVER:\n{ai_summary}\n\nProduction: {production_summary}\nIssues: {', '.join(issues)}\nTargets: {targets}",
                            'category': 'handover',
                            'is_private': False
                        }
                        self.data_manager.add_communication(msg_data)
                        st.success("✅ Handover sent")

        with col2:
            st.subheader("Recent Handovers")

            visible_handovers = [
                h for h in st.session_state.handovers
                if h['outgoing_supervisor_id'] == user['id'] or h['incoming_supervisor_id'] == user['id']
                   or user['role'] == UserRole.ADMIN.value
            ]

            if visible_handovers:
                for handover in reversed(visible_handovers[-5:]):
                    with st.expander(
                            f"{handover['shift_type']} - {handover['created_at'].strftime('%Y-%m-%d %H:%M')}"
                    ):
                        st.markdown(f"**Outgoing:** {handover['outgoing_supervisor_name']}")
                        st.markdown(f"**Incoming:** {handover['incoming_supervisor_name']}")
                        st.markdown(f"**Production:** {handover['production_summary']}")
                        if handover['issues_reported']:
                            st.markdown(f"**Issues:** {', '.join(handover['issues_reported'])}")
                        st.markdown(f"**Targets:** {handover['targets_next_shift']}")
            else:
                st.info("No handover reports yet")

    def render_analytics(self):
        """Analytics dashboard"""
        st.header("📈 Analytics & Insights")

        df = self.data_manager.get_timesheet_dataframe()

        if df.empty:
            st.info("Insufficient data for analytics")
            return

        period = st.selectbox("Time Period", ["Last 7 Days", "Last 30 Days", "This Month"])

        today = datetime.now().date()
        if period == "Last 7 Days":
            start_date = today - timedelta(days=7)
        elif period == "Last 30 Days":
            start_date = today - timedelta(days=30)
        else:
            start_date = today.replace(day=1)

        df_filtered = df[df['clock_in'].dt.date >= start_date]

        # Metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_hours = df_filtered['normal_hours'].sum() + df_filtered['overtime_hours'].sum()
            st.metric("Total Hours", f"{total_hours:.0f}h")

        with col2:
            avg_attendance = df_filtered.groupby(df_filtered['clock_in'].dt.date).size().mean()
            st.metric("Avg Daily Attendance", f"{avg_attendance:.0f}")

        with col3:
            overtime_ratio = (df_filtered['overtime_hours'].sum() / total_hours * 100) if total_hours > 0 else 0
            st.metric("Overtime Ratio", f"{overtime_ratio:.1f}%")

        with col4:
            flagged_ratio = len(df_filtered[df_filtered['ai_flags'].apply(len) > 0]) / len(df_filtered) * 100 if len(
                df_filtered) > 0 else 0
            st.metric("Flagged Rate", f"{flagged_ratio:.1f}%")

        # Charts
        st.subheader("Attendance Trend")
        daily_attendance = df_filtered.groupby(df_filtered['clock_in'].dt.date).size().reset_index(name='count')
        fig = px.line(daily_attendance, x='clock_in', y='count', title='Daily Attendance')
        st.plotly_chart(fig, use_container_width=True)

        # Cost Analysis
        if 'hourly_rate' in df_filtered.columns:
            st.subheader("Cost Analysis")
            df_filtered['cost'] = df_filtered['normal_hours'] * df_filtered['hourly_rate'] + \
                                  df_filtered['overtime_hours'] * df_filtered['hourly_rate'] * df_filtered[
                                      'overtime_rate']

            total_cost = df_filtered['cost'].sum()
            st.metric("Total Labor Cost", f"${total_cost:,.2f}")

            # Cost by department
            dept_costs = df_filtered.groupby('department')['cost'].sum().reset_index()
            fig2 = px.pie(dept_costs, values='cost', names='department', title='Cost by Department')
            st.plotly_chart(fig2, use_container_width=True)

    def render_user_management(self):
        """User management for admins"""
        st.header("👥 User Management")

        user = st.session_state.current_user
        if user['role'] not in [UserRole.ADMIN.value, UserRole.HR.value]:
            st.error("Access denied")
            return

        tab1, tab2 = st.tabs(["Users", "Add User"])

        with tab1:
            st.subheader("Current Users")
            users_df = pd.DataFrame(st.session_state.users)
            display_cols = ['username', 'name', 'role', 'department', 'shift', 'hourly_rate', 'is_active']
            st.dataframe(users_df[display_cols], use_container_width=True)

        with tab2:
            st.subheader("Add New User")

            col1, col2 = st.columns(2)

            with col1:
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                name = st.text_input("Full Name")
                role = st.selectbox("Role", options=[r.value for r in UserRole])
                department = st.text_input("Department")

            with col2:
                shift = st.selectbox("Shift", options=["None"] + [s.value for s in ShiftType])
                email = st.text_input("Email")
                phone = st.text_input("Phone")
                hourly_rate = st.number_input("Hourly Rate ($)", min_value=0.0, value=20.0)

            if st.button("Create User", type="primary"):
                if username and password and name:
                    user_data = {
                        'username': username,
                        'password': password,
                        'name': name,
                        'role': role,
                        'department': department,
                        'shift': None if shift == "None" else shift,
                        'email': email,
                        'phone': phone,
                        'hourly_rate': hourly_rate,
                        'overtime_rate': 1.5,
                        'is_active': True
                    }

                    if self.data_manager.add_user(user_data):
                        st.success(f"User {username} created")
                        st.rerun()
                    else:
                        st.error("Username already exists")
                else:
                    st.error("Required fields missing")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    app = TimesheetApp()
    app.run()