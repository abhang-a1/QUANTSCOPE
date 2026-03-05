"""
QuantScope AI - Authentication Routes
Handles user registration, login, and session management
"""

from flask import Blueprint, request, jsonify, render_template, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import sqlite3
import secrets
import datetime
import re

auth_bp = Blueprint('auth', __name__)

# Database configuration
DATABASE = 'quantscope_users.db'

def get_db():
    """Get database connection"""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize database with users table"""
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            is_active BOOLEAN DEFAULT 1,
            age_verified BOOLEAN DEFAULT 0,
            terms_accepted BOOLEAN DEFAULT 0
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            token TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    conn.commit()
    conn.close()

def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password(password):
    """
    Validate password strength
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters"

    if not re.search(r'[A-Z]', password):
        return False, "Password must contain an uppercase letter"

    if not re.search(r'[a-z]', password):
        return False, "Password must contain a lowercase letter"

    if not re.search(r'[0-9]', password):
        return False, "Password must contain a number"

    special_chars_pattern = r"[!@#$%^&*()_+=\[\]{}|;:',.<>?/\\-]"
    if not re.search(special_chars_pattern, password):
        return False, "Password must contain a special character (!@#$%^&* etc.)"

    return True, "Password is valid"

def generate_token():
    """Generate secure session token"""
    return secrets.token_urlsafe(32)

def login_required(f):
    """Decorator to require login for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization') or session.get('token')

        if not token:
            # API requests (JS fetch) -> return JSON 401
            if request.path.startswith('/api/'):
                return jsonify({'success': False, 'message': 'Login required'}), 401
            # Browser page requests -> redirect to login page
            return redirect('/login')

        # Verify token
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            'SELECT user_id FROM sessions WHERE token = ? AND expires_at > ?',
            (token, datetime.datetime.now())
        )
        session_data = cursor.fetchone()
        conn.close()

        if not session_data:
            # API call -> JSON 401
            if request.path.startswith('/api/'):
                return jsonify({'success': False, 'message': 'Invalid or expired session'}), 401
            # Browser -> redirect to login
            return redirect('/login')

        request.user_id = session_data['user_id']
        return f(*args, **kwargs)

    return decorated_function

# ============================================
# ROUTES
# ============================================

@auth_bp.route('/login')
def login_page():
    """Render login page"""
    return render_template('login.html')

@auth_bp.route('/signup')
def signup_page():
    """Render signup page"""
    return render_template('signup.html')

@auth_bp.route('/api/auth/signup', methods=['POST'])
def signup():
    """Handle user registration"""
    try:
        data = request.get_json()

        # Validate required fields
        required_fields = ['email', 'password', 'firstName', 'lastName', 'ageVerified', 'termsAccepted']
        for field in required_fields:
            if field not in data:
                return jsonify({'success': False, 'message': f'Missing field: {field}'}), 400

        email = data['email'].strip().lower()
        password = data['password']
        first_name = data['firstName'].strip()
        last_name = data['lastName'].strip()
        age_verified = data['ageVerified']
        terms_accepted = data['termsAccepted']

        if not validate_email(email):
            return jsonify({'success': False, 'message': 'Invalid email format'}), 400

        valid, message = validate_password(password)
        if not valid:
            return jsonify({'success': False, 'message': message}), 400

        if len(first_name) < 2 or len(last_name) < 2:
            return jsonify({'success': False, 'message': 'Names must be at least 2 characters'}), 400

        if not age_verified:
            return jsonify({'success': False, 'message': 'You must be 18 or older'}), 400

        if not terms_accepted:
            return jsonify({'success': False, 'message': 'You must accept the Terms & Conditions'}), 400

        conn = get_db()
        cursor = conn.cursor()

        cursor.execute('SELECT id FROM users WHERE email = ?', (email,))
        if cursor.fetchone():
            conn.close()
            return jsonify({'success': False, 'message': 'Email already registered'}), 400

        password_hash = generate_password_hash(password, method='pbkdf2:sha256')

        cursor.execute(
            '''INSERT INTO users (email, password_hash, first_name, last_name, 
                age_verified, terms_accepted) 
                VALUES (?, ?, ?, ?, ?, ?)''',
            (email, password_hash, first_name, last_name, age_verified, terms_accepted)
        )

        conn.commit()
        user_id = cursor.lastrowid
        conn.close()

        print(f"✅ User registered: {email} (ID: {user_id})")

        return jsonify({
            'success': True,
            'message': 'Account created successfully! Please log in.',
            'redirect': '/login'
        }), 201

    except Exception as e:
        print(f"❌ Signup error: {e}")
        return jsonify({'success': False, 'message': f'Registration failed: {str(e)}'}), 500

@auth_bp.route('/api/auth/login', methods=['POST'])
def login():
    """Handle user login"""
    try:
        data = request.get_json()

        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        remember = data.get('remember', False)

        if not email or not password:
            return jsonify({'success': False, 'message': 'Email and password required'}), 400

        conn = get_db()
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
        user = cursor.fetchone()

        if not user or not check_password_hash(user['password_hash'], password):
            conn.close()
            return jsonify({'success': False, 'message': 'Invalid email or password'}), 401

        if not user['is_active']:
            conn.close()
            return jsonify({'success': False, 'message': 'Account is disabled'}), 403

        token = generate_token()
        expires_at = datetime.datetime.now() + datetime.timedelta(days=30 if remember else 1)

        cursor.execute(
            'INSERT INTO sessions (user_id, token, expires_at) VALUES (?, ?, ?)',
            (user['id'], token, expires_at)
        )

        cursor.execute(
            'UPDATE users SET last_login = ? WHERE id = ?',
            (datetime.datetime.now(), user['id'])
        )

        conn.commit()
        conn.close()

        session['token'] = token
        session['user_id'] = user['id']
        session['user_email'] = user['email']
        session['user_name'] = f"{user['first_name']} {user['last_name']}"

        print(f"✅ User logged in: {email}")

        return jsonify({
            'success': True,
            'message': f'Welcome back, {user["first_name"]}!',
            'token': token,
            'redirect': '/dashboard',
            'user': {
                'id': user['id'],
                'email': user['email'],
                'firstName': user['first_name'],
                'lastName': user['last_name']
            }
        }), 200

    except Exception as e:
        print(f"❌ Login error: {e}")
        return jsonify({'success': False, 'message': f'Login failed: {str(e)}'}), 500

@auth_bp.route('/api/auth/logout', methods=['POST'])
def logout():
    """Handle user logout"""
    token = request.headers.get('Authorization') or session.get('token')
    user_email = session.get('user_email', 'Unknown')

    if token:
        try:
            conn = get_db()
            cursor = conn.cursor()
            cursor.execute('DELETE FROM sessions WHERE token = ?', (token,))
            conn.commit()
            conn.close()
            print(f"✅ User logged out: {user_email}")
        except Exception as e:
            print(f"⚠️ Logout cleanup error: {e}")

    session.clear()

    return jsonify({
        'success': True,
        'message': 'Logged out successfully',
        'redirect': '/login'
    }), 200

@auth_bp.route('/api/auth/me', methods=['GET'])
@login_required
def get_current_user():
    """Get current user info"""
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            'SELECT id, email, first_name, last_name, created_at, last_login FROM users WHERE id = ?',
            (request.user_id,)
        )
        user = cursor.fetchone()
        conn.close()

        if not user:
            return jsonify({'success': False, 'message': 'User not found'}), 404

        return jsonify({
            'success': True,
            'user': {
                'id': user['id'],
                'email': user['email'],
                'firstName': user['first_name'],
                'lastName': user['last_name'],
                'createdAt': user['created_at'],
                'lastLogin': user['last_login']
            }
        }), 200

    except Exception as e:
        print(f"❌ Get user error: {e}")
        return jsonify({'success': False, 'message': 'Failed to get user info'}), 500

try:
    init_db()
    print("✅ Authentication system initialized")
except Exception as e:
    print(f"❌ Database initialization error: {e}")

